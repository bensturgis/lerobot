
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.cm as cm
import torch
from torch import Tensor

from lerobot.common.policies.factory import make_flow_matching_uncertainty_scoring_metric
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.common.policies.flow_matching.uncertainty.configuration_uncertainty_sampler import (
    ComposedSequenceSamplerConfig,
)
from lerobot.common.policies.flow_matching.visualizers import NoiseToActionVisualizer
from lerobot.configs.default import NoiseToActionVisConfig

from ..configuration_flow_matching import FlowMatchingConfig
from .base_sampler import FlowMatchingUncertaintySampler


class ComposedSequenceSampler(FlowMatchingUncertaintySampler):
    """
    Samples action sequences, composes them with a previously executed sequence segment, 
    and evaluates their uncertainty under the flow matching model.

    The key idea is that if the composed sequences have a low uncertainty, then the model
    successfully anticipated what is likely to happen next. This implies a good internal model 
    of the environment.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: ComposedSequenceSamplerConfig,
        flow_matching_model: FlowMatchingModel,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
        """
        extra_sampling_times = cfg.scoring_metric.velocity_eval_times if (cfg.scoring_metric.metric_type == "inter_vel_diff") else None

        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            flow_matching_model=flow_matching_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            extra_sampling_times=extra_sampling_times,
        )
        self.method_name = "composed_sequence"

        # Initialize scoring metric
        self.scoring_metric = make_flow_matching_uncertainty_scoring_metric(
            config=cfg.scoring_metric,
            uncertainty_sampler=self,
        )

        # Sampler-specific settings
        self.cfg = cfg

        # Store the conditioning vector and ODE states from the previous action
        # sequence generation
        self.prev_global_cond: Optional[Tensor] = None
        self.prev_ode_states: Optional[Tensor] = None

        # Index of the selected action sequence from the previous actions batch
        self.prev_selected_action_idx: Optional[int] = None

        # ------------------------------- DEBUGGING VISUALIZATION --------------------------------
        # vis_config = NoiseToActionVisConfig()
        # now = dt.datetime.now()
        # vis_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_compsed_sequence_sampler_debugging"
        # output_dir = Path("outputs/visualizations") / vis_dir
        # self.visualizer = NoiseToActionVisualizer(
        #     cfg=vis_config,
        #     flow_matching_cfg=self.flow_matching_cfg,
        #     velocity_model=self.flow_matching_model.unet,
        #     output_root=output_dir,
        # )
        # ----------------------------------------------------------------------------------------
        
    def conditional_sample_with_uncertainty(
        self,
        observation: dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Samples num_action_seq_samples many new action sequences and computes uncertainty
        scores by composing them with a previous action sequence and evaluating them under
        the flow model.

        Args:
            observation: Info about the environment used to create the conditioning vector for
                the flow matching model. It has to contain the following items:
                {
                "observation.state": (B, n_obs_steps, state_dim)

                "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                    AND/OR
                "observation.environment_state": (B, environment_dim)
                }
            generator: PyTorch random number generator.

        Returns:
            - Action sequence samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores where a higher value means more uncertain.
                Shape: [num_action_seq_samples,].
        """
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.flow_matching_model.prepare_global_conditioning(observation)
        
        # Adjust shape of conditioning vector
        global_cond = self._prepare_conditioning(global_cond)

        # Sample noise priors        
        new_noise_sample = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        if self.prev_selected_action_idx is not None:
            # Reuse overlapping segment of noise from the previously selected trajectory
            # so that the newly sampled noise remains consistent with already executed actions
            new_noise_overlap_end = self.exec_start_idx + (self.horizon - self.exec_end_idx)
            prev_noise_sample = self.prev_ode_states[0, self.prev_selected_action_idx]
            prev_noise_sample_duplicated = prev_noise_sample.expand(
                self.num_action_seq_samples, -1, -1
            )
            new_noise_sample[:, self.exec_start_idx:new_noise_overlap_end, :] = prev_noise_sample_duplicated[:, self.exec_end_idx:, :]

        # Solve ODE forward from noise to sample action sequences
        new_ode_states = self.sampling_ode_solver.sample(
            x_0=new_noise_sample,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )

        # Store sampled action sequences for logging
        sampled_action_seqs = new_ode_states[-1]
        self.latest_action_candidates = sampled_action_seqs

        # If no previous trajectory is stored, return placeholder uncertainties
        if self.prev_selected_action_idx is None:
            uncertainty_scores = torch.full(
                (self.num_action_seq_samples,),
                float('-inf'),
                dtype=self.dtype,
                device=self.device
            )
        else:
            # Compose full ODE states from stored previous and new action generation
            composed_ode_states = self.compose_ode_states(
                prev_ode_states=self.prev_ode_states[
                    :, self.prev_selected_action_idx:self.prev_selected_action_idx+1, :, :
                ],
                new_ode_states=new_ode_states
            )

            # Broadcast the selected past ODE states so all new samples are compared against the same executed prefix
            prev_selected_ode_states = (
                self.prev_ode_states[:, self.prev_selected_action_idx:self.prev_selected_action_idx+1, :, :]
                    .expand(-1, self.num_action_seq_samples, -1, -1)
            )

            # Compute uncertainty based on selected metric
            if self.scoring_metric.name in ("terminal_vel_norm", "mode_distance", "likelihood"):
                uncertainty_scores = self.scoring_metric(
                    action_sequences=composed_ode_states[-1],
                    velocity_model=self.velocity_model,
                    global_cond=self.prev_global_cond,
                )
            elif self.scoring_metric.name == "inter_vel_diff":
                uncertainty_scores = self.scoring_metric(
                    ref_ode_states=prev_selected_ode_states,
                    ref_velocity_model=self.velocity_model,
                    ref_global_cond=self.prev_global_cond,
                    cmp_ode_states=composed_ode_states,
                    cmp_velocity_model=self.velocity_model,
                    cmp_global_cond=self.prev_global_cond,
                )
            else:
                raise ValueError(f"Unknown uncertainty metric: {self.scoring_metric.name}.")

            # ------------------------------- DEBUGGING VISUALIZATION --------------------------------
            # self.visualizer._update_run_dir()
            # combined_horizon = 2 * self.horizon - new_noise_overlap_end
            # colors = cm.get_cmap('plasma')(torch.arange(combined_horizon) / (combined_horizon - 1))
            # step_labels = ("t", *[f"t+{k}" for k in range(1, combined_horizon)])
            
            # prev_actions_overlay = {
            #     "label": "Previous ODE States",
            #     "ode_states": self.prev_ode_states.transpose(0, 1),
            #     "colors": colors[:self.horizon],
            #     # "step_labels": step_labels[:self.horizon],
            #     "text_kwargs": {"xytext": (-14, -12)},
            #     "scale": 40,
            # }
            # new_actions_overlay = {
            #     "label": "Current ODE States",
            #     "ode_states": new_ode_states.transpose(0, 1),
            #     "colors": colors[new_noise_overlap_end:],
            #     # "step_labels": step_labels[new_noise_overlap_end:],
            #     "text_kwargs": {"xytext": (2, 2)},
            #     "scale": 50,
            #     "marker": "x",
            # }
            # self.visualizer.plot_noise_to_action_overlays(
            #     action_overlays=[prev_actions_overlay, new_actions_overlay]
            # )
            # ----------------------------------------------------------------------------------------
            
        # Store computed uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        # Store conditioning vector and ODE states from the previous action sampling step
        self.prev_global_cond = global_cond
        self.prev_ode_states = new_ode_states

        return sampled_action_seqs, uncertainty_scores
    