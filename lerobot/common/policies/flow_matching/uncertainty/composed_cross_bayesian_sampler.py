from typing import Optional, Tuple

import torch
from torch import Tensor

from lerobot.common.policies.factory import make_flow_matching_uncertainty_scoring_metric
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.common.policies.flow_matching.uncertainty.configuration_uncertainty_sampler import (
    CrossBayesianSamplerConfig,
)
from lerobot.common.policies.flow_matching.uncertainty.laplace_utils import (
    draw_laplace_flow_matching_model,
)
from lerobot.common.policies.flow_matching.uncertainty.scorer_artifacts import (
    ScorerArtifacts,
)

from ..configuration_flow_matching import FlowMatchingConfig
from .base_uncertainty_sampler import FlowMatchingUncertaintySampler


class ComposedCrossBayesianSampler(FlowMatchingUncertaintySampler):
    """
    Splices newly sampled action sequence tails onto the previously executed prefix and evaluates
    the full trajectories with a flow matching "scorer". The "scorer" model can be either an 
    independently trained ensemble model or a Laplace posterior draw. Uncertainty can be measured
    using several different metrics.

    The class therefore mixes
    - sequence composition from ComposedSequenceSampler and  
    - cross bayesian epistemic scoring from CrossBayesianSampler.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: CrossBayesianSamplerConfig,
        sampler_model: FlowMatchingModel,
        scorer_artifacts: ScorerArtifacts,
    ):
        """
        Initializes the composed sequence cross bayesian sampler.
        
        Args:
            cfg: Sampler-specific settings.
            sampler_model: The full flow matching model including velocity and RGB encoder.
            ensemble_model: An auxiliary flow matching model used for scoring when the scorer
                type is "ensemble".
            laplace_posterior: A Laplace approximation posterior used for scoring when the
                scorer type is "laplace".
        """
        extra_sampling_times = cfg.scoring_metric.velocity_eval_times if (cfg.scoring_metric.metric_type == "inter_vel_diff") else None

        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            flow_matching_model=sampler_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            extra_sampling_times=extra_sampling_times,
        )
        self.method_name = "composed_cross_bayesian"

        # Initialize scoring metric
        self.scoring_metric = make_flow_matching_uncertainty_scoring_metric(
            config=cfg.scoring_metric,
            uncertainty_sampler=self,
        )
        
        self.ensemble_model = scorer_artifacts.ensemble_model
        self.laplace_posterior = scorer_artifacts.laplace_posterior
        if cfg.scorer_type == "ensemble" and self.ensemble_model is None:
            raise ValueError("ensemble_model is required for scorer_type='ensemble'.")
        elif cfg.scorer_type == "laplace" and self.laplace_posterior is None:
            raise ValueError("laplace_posterior is required for scorer_type='laplace'.")
        elif cfg.scorer_type not in {"ensemble", "laplace"}:
            raise ValueError(f"Unknown scorer_type: {cfg.scorer_type!r}")
        
        # Sampler-specific settings
        self.cfg = cfg

        # Store the conditioning vectors and ODE states from the previous action
        # sequence generation
        self.prev_global_cond: Optional[Tensor] = None
        self.prev_ode_states: Optional[Tensor] = None
        self.prev_scorer_global_cond: Optional[Tensor] = None

       # Index of the selected action sequence from the previous actions batch
        self.prev_selected_action_idx: Optional[int] = None

        # Store scorer flow matching model from the previous action sequence generation
        self.prev_scorer_flow_matching_model: Optional[FlowMatchingModel] = None

    def conditional_sample_with_uncertainty(
        self,
        observation: dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, float]:
        """
        Sample candidate action sequences with the sampler flow-matching model, splice newly
        sampled tails onto the executed prefix, and score them with the configured epistemic
        scorer (ensemble or Laplace).

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
            - Action sequences drawn from the sampler model.
              Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty score where a higher value means more uncertain.      
        """
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.flow_matching_model.prepare_global_conditioning(observation)
        
        # Adjust shape of conditioning vector
        global_cond = self._reshape_conditioning(global_cond)

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
        self.latest_action_candidates = new_ode_states[-1]

        if self.cfg.scorer_type == "laplace":
            # Draw flow matching model from the Laplace posterior
            scorer_flow_matching_model = draw_laplace_flow_matching_model(
                laplace_posterior=self.laplace_posterior,
                flow_matching_model=self.flow_matching_model,
                generator=generator
            )
        else:
            scorer_flow_matching_model = self.ensemble_model

        # Create and prepare the scorer conditioning vector
        scorer_global_cond = scorer_flow_matching_model.prepare_global_conditioning(observation)
        scorer_global_cond = self._reshape_conditioning(scorer_global_cond)  # (B, global_cond_dim)

        if self.prev_selected_action_idx is None:
            # If no previous trajectory is stored, return placeholder uncertainty
            self.latest_uncertainty = float('-inf')
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
                    velocity_model=scorer_flow_matching_model.unet,
                    global_cond=self.prev_scorer_global_cond,
                )
            elif self.scoring_metric.name == "inter_vel_diff":
                uncertainty_scores = self.scoring_metric(
                    ref_ode_states=prev_selected_ode_states,
                    ref_velocity_model=self.velocity_model,
                    ref_global_cond=self.prev_global_cond,
                    cmp_ode_states=composed_ode_states,
                    cmp_velocity_model=self.prev_scorer_flow_matching_model.unet,
                    cmp_global_cond=self.prev_scorer_global_cond,
                )
            else:
                raise ValueError(f"Unknown uncertainty metric: {self.scoring_metric.name}.")

            # Average uncertainty scores and store for logging
            self.latest_uncertainty = uncertainty_scores.mean().item()

        # Store scorer model, conditioning vectors, ODE states from the previous action sampling step
        self.prev_scorer_flow_matching_model = scorer_flow_matching_model
        self.prev_scorer_global_cond = scorer_global_cond
        self.prev_global_cond = global_cond
        self.prev_ode_states = new_ode_states
        
        return self.latest_action_candidates, self.latest_uncertainty