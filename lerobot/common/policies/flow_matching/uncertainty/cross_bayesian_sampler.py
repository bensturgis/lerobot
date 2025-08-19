from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lerobot.common.policies.factory import make_flow_matching_uncertainty_scoring_metric
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.common.policies.flow_matching.uncertainty.configuration_uncertainty_sampler import (
    CrossBayesianSamplerConfig,
)
from lerobot.common.policies.flow_matching.uncertainty.laplace_utils import (
    draw_laplace_flow_matching_model,
    get_laplace_posterior,
)

from ..configuration_flow_matching import FlowMatchingConfig
from .base_sampler import FlowMatchingUncertaintySampler


class CrossBayesianSampler(FlowMatchingUncertaintySampler):
    """
    Samples action sequences from a "sampler" flow-matching model and evaluates their
    uncertainty under a separate "scorer" flow-matching model. The "scorer" model can be
    either an independently trained ensemble model or a Laplace posterior draw.
    Uncertainty can be measured using several different metrics.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: CrossBayesianSamplerConfig,
        sampler_flow_matching_model: FlowMatchingModel,
        ensemble_flow_matching_model: Optional[FlowMatchingModel] = None,
        laplace_calib_loader: Optional[DataLoader] = None,
        laplace_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
            sampler_flow_matching_model: The full flow matching model including velocity and RGB encoder.
            ensemble_flow_matching_model: Model to score sampled actions.
            laplace_calib_loader: DataLoader providing samples for fitting the Laplace approximation.
            laplace_path: Path to save or load the Laplace posterior.
        """
        extra_sampling_times = cfg.scoring_metric.velocity_eval_times if (cfg.scoring_metric.metric_type == "inter_vel_diff") else None

        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            flow_matching_model=sampler_flow_matching_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            extra_sampling_times=extra_sampling_times,
        )
        self.method_name = "cross_bayesian"

        # Initialize scoring metric
        self.scoring_metric = make_flow_matching_uncertainty_scoring_metric(
            config=cfg.scoring_metric,
            uncertainty_sampler=self,
        )
        
        if cfg.scorer_type == "ensemble":
            if ensemble_flow_matching_model is None:
                raise ValueError("ensemble_flow_matching_model is required for scorer_type='ensemble'.")
            self.ensemble_flow_matching_model = ensemble_flow_matching_model
        elif cfg.scorer_type == "laplace":
            if laplace_calib_loader is None and (laplace_path is None or not laplace_path.exists()):
                raise ValueError(
                    "scorer_type='laplace' requires either an existing laplace_path "
                    "or a laplace_calib_loader to fit a new posterior."
                )
            self.laplace_posterior = get_laplace_posterior(
                cfg=cfg,
                flow_matching_model=sampler_flow_matching_model,
                laplace_calib_loader=laplace_calib_loader,
                laplace_path=laplace_path,
            )
        else:
            raise ValueError(f"Unknown scorer_type: {cfg.scorer_type!r}")
        
        # Sampler-specific settings
        self.cfg = cfg

    def conditional_sample_with_uncertainty(
        self,
        observation: dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generates action sequences using a sampler flow matching model, then scores these
        samples under a Laplace-sampled or an ensemble model to obtain epistemic uncertainty.

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
            - sampled_action_seqs: Action sequences drawn from the sampler model.
              Shape: [num_action_seq_samples, horizon, action_dim].
            - uncertainty_scores: Uncertainty scores where a higher value means more
              uncertain. Shape: [num_action_seq_samples,].      
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
        ode_states = self.sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )

        # Store sampled action sequences for logging
        sampled_action_seqs = ode_states[-1]
        self.latest_action_candidates = sampled_action_seqs

        if self.cfg.scorer_type == "laplace":
            # Draw flow matching model from the Laplace posterior
            scorer_flow_matching_model = draw_laplace_flow_matching_model(
                laplace_posterior=self.laplace_posterior,
                flow_matching_model=self.flow_matching_model,
                generator=generator
            )
        else:
            scorer_flow_matching_model = self.ensemble_flow_matching_model

        # Create and prepare the scorer conditioning vector
        scorer_global_cond = scorer_flow_matching_model.prepare_global_conditioning(observation)
        scorer_global_cond = self._prepare_conditioning(scorer_global_cond)  # (B, global_cond_dim)

        # Compute uncertainty based on selected metric
        if self.scoring_metric.name in ("terminal_vel_norm", "mode_distance", "likelihood"):
            uncertainty_scores = self.scoring_metric(
                action_sequences=sampled_action_seqs,
                velocity_model=scorer_flow_matching_model.unet,
                global_cond=scorer_global_cond,
            )
        elif self.scoring_metric.name == "inter_vel_diff":
            uncertainty_scores = self.scoring_metric(
                ref_ode_states=ode_states,
                ref_velocity_model=self.velocity_model,
                ref_global_cond=global_cond,
                cmp_ode_states=ode_states,
                cmp_velocity_model=scorer_flow_matching_model.unet,
                cmp_global_cond=scorer_global_cond,
            )
        else:
            raise ValueError(f"Unknown uncertainty metric: {self.scoring_metric.name}.")

        # Store uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores
        
        return sampled_action_seqs, uncertainty_scores