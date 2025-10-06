import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.conditional_probability_path import (
    OTCondProbPath,
    VPDiffusionCondProbPath,
)

from ..uncertainty_adapters.uncertainty_adapter import UncertaintyModelAdapter
from .configuration_uncertainty_sampler import EntropySamplerConfig
from .uncertainty_sampler import UncertaintySampler


class EntropySampler(UncertaintySampler):
    """
    Estimates the terminal differential entropy H(p_1) of a flow matching model by
    integrating a time-dependent expectation formed from the velocity field u_t and
    state x_t, then adding the base entropy H(p_0) of the Gaussian prior.
    """
    def __init__(
        self,
        config: EntropySamplerConfig,
        model: UncertaintyModelAdapter,
    ):
        """
        Initializes the entropy sampler.

        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            model=model,
            num_action_samples=config.num_action_samples,
        )
        self.method_name = "entropy"

        # Sampler-specific settings
        self.config = config

        # Select conditional probability path used to compute time scalings
        self.cond_vf_type = self.model.cond_vf_config["type"]
        if self.cond_vf_type == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=self.model.cond_vf_config["beta_min"],
                beta_max=self.model.cond_vf_config["beta_max"],
            )
        elif self.cond_vf_type == "ot":
            self.cond_prob_path = OTCondProbPath(
                sigma_min=self.model.cond_vf_config["sigma_min"]
            )
        else:
            raise ValueError(
                f"Unknown conditional vector field type {self.cond_vf_type}."
            )

    def _get_scaling_factors(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute time-dependent factors used in the entropy-rate integrand.

        Args:
            t: Time grid. Shape: (timesteps,).

        Returns:
            For VP: -(2 / β(1-t)), 0.5 * β(1-t)
            For OT: -t/(1-t), 1/t
        """
        if self.cond_vf_type == "vp":
            beta = self.cond_prob_path.get_beta(t)
            return -(2.0 / beta), 0.5 * beta
        elif self.cond_vf_type == "ot":
            return t / (t - 1), 1.0 / t
        else:
            raise ValueError(
                "No entropy scaling factors provided for conditional " \
                f"VF type: {self.cond_vf_type}."
            )

    def _integrate_over_time(self, values: Tensor, time_grid: Tensor) -> float:
        """
        Integrate a 1D time series over [0, 1] with:
          - right rule on the first timestep (avoid t=0),
          - trapezoid on interior intervals,
          - left rule on final timestep (avoid t=1).

        Args:
            values: Integrand per time step. Shape: (timesteps,).
            time_grid: Increasing time grid including 0 and 1. Shape: (timesteps,).

        Returns:
            Integral value over [0, 1].
        """
        time_intervals = time_grid[1:] - time_grid[:-1]

        # Right rule on first interval
        first = values[1] * time_intervals[0]

        # Trapezoid rule on interior intervals
        interior = 0.5 * (values[1:-2] + values[2:-1]) * time_intervals[1:-1]

        # Left rule on last interval
        last = values[-2] * time_intervals[-1]

        return float(first + interior.sum() + last)

    def conditional_sample_with_uncertainty(
        self,
        observation: dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Tuple[Tensor, float]:
        """
        Generate action sequence candidates with the sampler model and estimate the terminal
        differential entropy H(p_1).

        Args:
            observation: Info about the environment used to create the conditioning for
                the flow matching model.
            generator: PyTorch random number generator.

        Returns:
            - Action sequence samples. Shape: [num_action_samples, horizon, action_dim].
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
        ode_states, velocities = self.sampling_ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
            return_intermediate_vels=True
        )

        # Build the integrand and average over samples at each time step
        scale_inv, alpha_ratio = self._get_scaling_factors(self.sampling_time_grid)

        integrand = scale_inv * (torch.norm(velocities, dim=(2, 3)) ** 2 - alpha_ratio.view(-1, 1) * (velocities * ode_states).sum(dim=(2, 3))).mean(dim=1)

        # Integrate over time
        entropy_path = self._integrate_over_time(values=integrand, time_grid=self.sampling_time_grid)

        # Compute and add base entropy
        entropy_gaussian = 0.5 * self.horizon * self.action_dim * (1.0 + math.log(2.0 * math.pi))

        total_entropy = entropy_gaussian + entropy_path

        # Store sampled action sequences and uncerainty for logging
        self.latest_uncertainty = total_entropy

        # Pick one action sequence at random
        actions, _ = self.rand_pick_action(action_candidates=ode_states[-1])

        return actions.to(device="cpu", dtype=torch.float32), self.latest_uncertainty

    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        pass
