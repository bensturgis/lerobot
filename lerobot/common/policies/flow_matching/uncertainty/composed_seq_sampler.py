
from typing import Tuple

import torch
from torch import Tensor

from lerobot.common.policies.factory import make_flow_matching_uncertainty_scoring_metric
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.common.policies.flow_matching.uncertainty.configuration_uncertainty_sampler import (
    ComposedSequenceSamplerConfig,
)

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

        # Store the action sequence and conditioning vector from the previous action
        # sequence generation
        self.prev_action_sequence = None
        self.prev_global_cond = None

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
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        new_action_seq = self.sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
        )
        # Store sampled action sequences for logging
        self.latest_action_candidates = new_action_seq

        # If no previous trajectory is stored, return placeholder uncertainties
        if self.prev_action_sequence is None:
            uncertainty_scores = torch.full(
                (self.num_action_seq_samples,),
                float('-inf'),
                dtype=self.dtype,
                device=self.device
            )
        else:
            # Compose full action sequences from stored prefix and newly sampled action sequences
            composed_action_seq = self.compose_action_seqs(
                prev_action_seq=self.prev_action_sequence,
                new_action_seq=new_action_seq  
            )

            # Compute uncertainty based on selected metric
            uncertainty_scores = self.scoring_metric(
                scorer_velocity_model=self.flow_matching_model.unet,
                scorer_global_cond=self.prev_global_cond,
                ode_states=composed_action_seq.unsqueeze(0),
                sampler_global_cond=global_cond,
                generator=generator,
            )      

        # Store computed uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        # Store conditioning vector of the scoring model from the previous action sampling step
        self.prev_global_cond = global_cond

        return new_action_seq, uncertainty_scores
    