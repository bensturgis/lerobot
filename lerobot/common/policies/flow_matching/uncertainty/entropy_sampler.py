import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..conditional_probability_path import OTCondProbPath, VPDiffusionCondProbPath
from ..configuration_flow_matching import FlowMatchingConfig
from ..modelling_flow_matching import FlowMatchingModel
from .base_uncertainty_sampler import FlowMatchingUncertaintySampler
from .configuration_uncertainty_sampler import EntropySamplerConfig


class EntropySampler(FlowMatchingUncertaintySampler):
    """
    Estimates the terminal differential entropy H(p₁) of a flow-matching model by
    integrating a time-dependent expectation formed from the velocity field u_t and
    state x_t, then adding the base entropy H(p₀) of the Gaussian prior. The time
    integral is computed on a fixed grid and avoids endpoint singularities with
    one-sided rules at t=0 and t=1.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: EntropySamplerConfig,
        flow_matching_model: FlowMatchingModel,
    ):
        """
        Initializes the entropy sampler.

        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            flow_matching_model=flow_matching_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
        )
        self.method_name = "entropy"

        # Sampler-specific settings
        self.cfg = cfg

        # Select conditional probability path used to compute time scalings
        self.cond_vf_type = self.flow_matching_cfg.cond_vf_type
        if self.cond_vf_type == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=self.flow_matching_cfg.beta_min,
                beta_max=self.flow_matching_cfg.beta_max,
            )
        elif self.cond_vf_type == "ot":
            self.cond_prob_path = OTCondProbPath()
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
        differential entropy H(p₁).

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
            - Uncertainty score where a higher value means more uncertain.
        """
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.flow_matching_model.prepare_global_conditioning(observation)
        
        # Adjust shape of conditioning vector
        global_cond = self._reshape_conditioning(global_cond)

        # Sample noise priors
        noise_sample = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states, velocities = self.sampling_ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
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
        self.latest_action_candidates = ode_states[-1]
        self.latest_uncertainty = total_entropy

        return self.latest_action_candidates, self.latest_uncertainty