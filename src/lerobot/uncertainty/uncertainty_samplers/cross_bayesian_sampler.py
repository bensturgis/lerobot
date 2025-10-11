from typing import Optional, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.factory import make_uncertainty_scoring_metric

from ..uncertainty_scoring.laplace_utils.posterior_builder import sample_adapter_from_posterior
from ..uncertainty_scoring.scorer_artifacts import ScorerArtifacts
from .configuration_uncertainty_sampler import (
    CrossBayesianSamplerConfig,
)
from .uncertainty_sampler import UncertaintySampler


class CrossBayesianSampler(UncertaintySampler):
    """
    Samples action sequences from a "sampler" flow-matching model and evaluates their
    uncertainty under a separate "scorer" flow-matching model. The "scorer" model can be
    either an independently trained ensemble model or a Laplace posterior draw.
    Uncertainty can be measured using several different metrics.
    """
    def __init__(
        self,
        config: CrossBayesianSamplerConfig,
        sampler_model: BaseFlowMatchingAdapter,
        scorer_artifacts: ScorerArtifacts,
    ):
        """
        Initializes the Initializes the cross bayesian sampler.

        Args:
            config: Sampler-specific settings.
            sampler_model: The flow matching model using for sampling actions.
            scorer_artifacts: Artifacts required by the scorer. Provide exactly one matching the configured scorer type.
                - ensemble_adapter: Adapter that wraps auxiliary flow matching model used for scoring when the scorer
                    type is "ensemble".
                - laplace_posterior: A Laplace approximation posterior used for scoring when the scorer type is "laplace".
        """
        extra_sampling_times = config.scoring_metric.velocity_eval_times if (config.scoring_metric.metric_type == "inter_vel_diff") else None

        super().__init__(
            model=sampler_model,
            num_action_samples=config.num_action_samples,
            extra_sampling_times=extra_sampling_times,
        )
        self.method_name = "cross_bayesian"

        # Initialize scoring metric
        self.scoring_metric = make_uncertainty_scoring_metric(
            config=config.scoring_metric,
            uncertainty_sampler=self,
        )

        self.ensemble_adapter = scorer_artifacts.ensemble_adapter
        self.laplace_posterior = scorer_artifacts.laplace_posterior
        if config.scorer_type == "ensemble" and self.ensemble_adapter is None:
            raise ValueError("ensemble_adapter is required for scorer_type='ensemble'.")
        elif config.scorer_type == "laplace" and self.laplace_posterior is None:
            raise ValueError("laplace_posterior is required for scorer_type='laplace'.")
        elif config.scorer_type not in {"ensemble", "laplace"}:
            raise ValueError(f"Unknown scorer_type: {config.scorer_type!r}")
        self.scorer_model = None

        # Sampler-specific settings
        self.config = config

    def conditional_sample_with_uncertainty(
        self,
        observation: dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, float]:
        """
        Generates action sequences using a sampler flow matching model, then scores these
        samples under a Laplace-sampled or an ensemble model, and finally averages these scores
        to obtain an epistemic uncertainty meaure.

        Args:
            observation: Info about the environment used to create the conditioning for
                the flow matching model.
            generator: PyTorch random number generator.

        Returns:
            - Action sequences drawn from the sampler model.
              Shape: [num_action_samples, horizon, action_dim].
            - Uncertainty score where a higher value means more uncertain.
        """
        # Build the velocity function conditioned on the current observation
        conditioning = self.model.prepare_conditioning(observation, self.num_action_samples)
        velocity_fn = self.model.make_velocity_fn(conditioning=conditioning)

        # Sample noise priors
        noise_sample = self.model.sample_prior(
            num_samples=self.num_action_samples,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states = self.sampling_ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )

        if self.config.scorer_type == "laplace":
            # Draw flow matching model from the Laplace posterior
            self.scorer_model = sample_adapter_from_posterior(
                laplace_posterior=self.laplace_posterior,
                uncertainty_adapter=self.model,
                generator=generator
            )
        else:
            self.scorer_model = self.ensemble_adapter

        # Build the conditioned velocity function of the scorer
        scorer_conditioning = self.scorer_model.prepare_conditioning(observation, self.num_action_samples)
        scorer_velocity_fn = self.scorer_model.make_velocity_fn(conditioning=scorer_conditioning)

        # Compute uncertainty based on selected metric
        if self.scoring_metric.name in ("terminal_vel_norm", "mode_distance", "likelihood"):
            uncertainty_scores = self.scoring_metric(
                action_sequences=ode_states[-1],
                velocity_fn=scorer_velocity_fn,
            )
        elif self.scoring_metric.name == "inter_vel_diff":
            uncertainty_scores = self.scoring_metric(
                ref_ode_states=ode_states,
                ref_velocity_fn=velocity_fn,
                cmp_ode_states=ode_states,
                cmp_velocity_fn=scorer_velocity_fn,
            )
        else:
            raise ValueError(f"Unknown uncertainty metric: {self.scoring_metric.name}.")

        # Average uncertainty scores and store for logging
        self.uncertainty = uncertainty_scores.mean().item()

        # Pick one action sequence at random
        self.action_candidates = ode_states[-1]
        actions, _ = self.rand_pick_action(action_candidates=self.action_candidates)

        return actions.to(device="cpu", dtype=torch.float32), self.uncertainty

    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        pass
