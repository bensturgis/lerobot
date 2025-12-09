from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.distributions import Independent, Normal

from lerobot.policies.common.flow_matching.conditional_probability_path import (
    OTCondProbPath,
    VPDiffusionCondProbPath,
)
from lerobot.policies.common.flow_matching.ode_solver import (
    ODESolver,
    make_lik_estimation_time_grid,
    select_ode_states,
)

from ..uncertainty_samplers.configuration_uncertainty_sampler import ScoringMetricConfig
from ..uncertainty_samplers.uncertainty_sampler import UncertaintySampler


class UncertaintyMetric(ABC):  # noqa: B024
    """Abstract base class for uncertainty metrics."""
    name: str = "base"
    type: str = "base"


class TerminalStateMetric(UncertaintyMetric, ABC):
    """
    Abstract base class for uncertainty metrics that operate on terminal action sequences,
    i.e., the final outputs of the flow matching ODE integration.
    Subclasses define how the velocity model is applied to score these sequences and produce
    an uncertainty score.
    """
    type: str = "terminal"

    @abstractmethod
    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute an uncertainty score for a batch of action sequences using a scorer flow matching model.
        Args:
            velocity_fn: Velocity function defining the right-hand side of the flow matching ODE
                d/dt φ_t(x) = v_t(φ_t(x), conditoning).
            action_sequence: Final action sequences to score. Shape: (batch_size,
                horizon, action_dim).

        Returns:
            Uncertainty scores per action sequence where larger values indicate higher uncertainty.
            Shape: (batch_size,).
        """
        raise NotImplementedError


class TerminalVelNorm(TerminalStateMetric):
    """
    Average L2 norm of scorer velocities evaluated on the terminal action sequence
    across specified evaluation times.
    """
    name: str = "terminal_vel_norm"

    def __init__(self, config: ScoringMetricConfig):
        """
        Args:
            config: Scoring metric settings.
        """
        self.velocity_eval_times = config.velocity_eval_times

    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        **_: Any
    ) -> Tensor:
        """
        Evaluate the velocity only at the terminal action sequences for multiple evaluation
        times and return the mean L2 norm as the uncertainty score.
        """
        # Evaluate velocity on the final sampled sequence
        terminal_vel_norms: list[float] = []
        for time in self.velocity_eval_times:
            velocity = velocity_fn(
                x_t=action_sequences, t=torch.tensor(time, device=action_sequences.device)
            )
            terminal_vel_norms.append(torch.norm(velocity, dim=(1, 2)))

        # Use average velocity norm as uncertainty score
        return torch.stack(terminal_vel_norms, dim=0).mean(dim=0)


class ModeDistance(TerminalStateMetric):
    """
    Estimates "distance to the next mode" by averaging (1 - t) · ‖v(x; t)‖ over specified
    evaluation times.
    """
    name: str = "mode_distance"

    def __init__(self, config: ScoringMetricConfig):
        """
        Args:
            config: Scoring metric settings.
            device: The PyTorch device on which to store and perform tensor operations.
        """
        self.velocity_eval_times = config.velocity_eval_times

    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        **_: Any,
    ) -> Tensor:
        """
        Compute the proxy "distance-from-mode" score computed by averaging (1 - t) * ‖v(x; t)‖
        of the velocity at the terminal action sequence x across the specified evaluation times.
        """
        distances: list[Tensor] = []
        # Loop over each time in [0, 1) at which we want to probe the velocity field
        for time in self.velocity_eval_times:
            # Query the velocity field at the terminal action sequence and this time
            velocity = velocity_fn(
                x_t=action_sequences, t=torch.tensor(time, device=action_sequences.device)
            )
            velocity_norm = torch.norm(velocity, dim=(1, 2))
            # Scale by (1 - time) as a simple proxy for “distance from the mode”
            # (i.e. how far a particle would still travel under constant velocity)
            distance = (1 - time) * velocity_norm
            distances.append(distance)

        return torch.stack(distances, dim=0).mean(dim=0)


class Likelihood(TerminalStateMetric):
    """
    Uncertainty metric that scores an action sequence by its negative log-likelihood under a scorer model.
    Uses an ODE solver to estimate likelihood along a reverse-time trajectory.
    """
    name: str = "likelihood"

    def __init__(self, config: ScoringMetricConfig, uncertainty_sampler: UncertaintySampler):
        """
        Args:
            config: Scoring metric settings.
        """
        self.device = uncertainty_sampler.device
        self.dtype = uncertainty_sampler.dtype
        # Noise distribution is an isotropic Gaussian
        horizon = uncertainty_sampler.horizon
        action_dim = uncertainty_sampler.action_dim
        self.gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(horizon, action_dim, device=self.device, dtype=self.dtype),
                scale = torch.ones(horizon, action_dim, device=self.device, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob

        # ODE solver settings for likelihood estimation
        self.lik_ode_solver_cfg = config.likelihood_ode_solver_cfg

        # Build time grid for likelihood estimation based on ODE solver method
        self.lik_estimation_time_grid = make_lik_estimation_time_grid(
            ode_solver_method=self.lik_ode_solver_cfg.method,
            device=self.device,
            dtype=self.dtype,
        )

        self.ode_solver = ODESolver()

    def __call__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        action_sequences: Tensor,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> Tensor:
        """
        Run a reverse-time ODE under the scorer model to compute the log-likelihood of the
        action sequence; the score is the negative log-likelihood.
        """
        # Compute log-likelihood of sampled action sequences in scorer model
        _, log_probs = self.ode_solver.sample_with_log_likelihood(
            x_init=action_sequences,
            time_grid=self.lik_estimation_time_grid,
            velocity_fn=velocity_fn,
            log_p_0=self.gaussian_log_density,
            method=self.lik_ode_solver_cfg.method,
            atol=self.lik_ode_solver_cfg.atol,
            rtol=self.lik_ode_solver_cfg.rtol,
            exact_divergence=self.lik_ode_solver_cfg.exact_divergence,
            generator=generator,
        )

        # Use negative log-likelihood as uncertainty score
        return -log_probs


class InterVelDiff(UncertaintyMetric):
    """
    Uncertainty metric based on intermediate velocity discrepancies.

    Compares the velocities predicted by two flow matching models along their
    respective ODE trajectories. Large values indicate stronger disagreement
    between the reference and comparison models.
    """
    name: str = "inter_vel_diff"
    type: str = "trajectory"

    def __init__(self, config: ScoringMetricConfig, uncertainty_sampler: UncertaintySampler):
        """
        Args:
            config: Scoring metric settings.
        """
        self.velocity_eval_times = config.velocity_eval_times
        self.device = uncertainty_sampler.device
        self.dtype = uncertainty_sampler.dtype
        self.ode_solver = uncertainty_sampler.sampling_ode_solver
        self.sampling_time_grid = uncertainty_sampler.sampling_time_grid
        self.cond_vf_type = uncertainty_sampler.cond_vf_config["type"]
        if self.cond_vf_type == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=uncertainty_sampler.cond_vf_config["beta_min"],
                beta_max=uncertainty_sampler.cond_vf_config["beta_max"],
            )
        elif self.cond_vf_type == "ot":
            self.cond_prob_path = OTCondProbPath(uncertainty_sampler.cond_vf_config["sigma_min"])
        else:
            raise ValueError(
                f"Unknown conditional vector field type {self.cond_vf_type}."
            )

    def __call__(
        self,
        ref_ode_states: Tensor,
        ref_velocity_fn: Callable[[Tensor, Tensor], Tensor],
        cmp_ode_states: Tensor,
        cmp_velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """
        Compute the average velocity discrepancy between two trajectories at intermediate ODE states.

        For each evaluation time t, this metric computes the L2 difference between the velocity
        predicted by the reference model at x_ref(t) and the comparison model at x_cmp(t):

            ||v_ref(x_ref(t), t) - v_cmp(x_cmp(t), t)||

        The discrepancies are integrated over time.

        Args:
            ref_ode_states: ODE integration states of the reference trajectory.
                Shape: (timesteps, batch_size, horizon, action_dim).
            ref_velocity_fn: Conditional velocity function associated with the reference trajectory.
            ref_global_cond: Conditioning vector for the reference model.
                Shape: (batch_size, cond_dim).
            cmp_ode_states: ODE integration states of the comparison trajectory.
                Shape: (timesteps, batch_size, horizon, action_dim).
            cmp_velocity_fn: Conditional velocity function associated with the comparison trajectory.
            cmp_global_cond: Conditioning vector for the comparison model.
                Shape: (batch_size, cond_dim).

        Returns:
            Uncertainty scores per trajectory sample, where larger values indicate stronger
            disagreement between reference and comparison velocities. Shape: (batch_size,).
        """
        # Select the ODE states that correspond to the velocity evaluation times
        selected_ref_ode_states, selected_ref_grid_times = select_ode_states(
            time_grid=self.sampling_time_grid,
            ode_states=ref_ode_states,
            requested_times=torch.tensor(self.velocity_eval_times, device=self.device, dtype=self.dtype)
        )
        selected_cmp_ode_states, selected_cmp_grid_times = select_ode_states(
            time_grid=self.sampling_time_grid,
            ode_states=cmp_ode_states,
            requested_times=torch.tensor(self.velocity_eval_times, device=self.device, dtype=self.dtype)
        )
        if not torch.equal(selected_ref_grid_times, selected_cmp_grid_times):
            raise ValueError(
                f"Mismatch in evaluation times: reference times {selected_ref_grid_times.tolist()} "
                f"vs comparison times {selected_cmp_grid_times.tolist()}."
            )

        # Evaluate velocity difference between reference and comparison trajectory at each intermediate time point
        batch_size = ref_ode_states.shape[1]
        inter_vel_diff_score: Tensor = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        for idx, (time, ref_inter_state, cmp_inter_state) in enumerate(zip(
            selected_ref_grid_times, selected_ref_ode_states, selected_cmp_ode_states, strict=False
        )):
            # Determine dt: difference to next time or to 1.0 for last step
            dt = selected_ref_grid_times[idx + 1] - time if idx < len(selected_ref_grid_times) - 1 else 1.0 - time

            ref_velocity = ref_velocity_fn(x_t=ref_inter_state, t=torch.tensor(time, device=self.device))
            cmp_velocity = cmp_velocity_fn(x_t=cmp_inter_state, t=torch.tensor(time, device=self.device))
            # L2 norm across horizon and action dims gives magnitude of velocity difference
            velocity_difference = torch.norm(ref_velocity - cmp_velocity, dim=(1, 2)) ** 2

            # Scale velocity difference by factor that depends on conditional vector field type
            inter_vel_diff_score += (
                self.cond_prob_path.get_vel_diff_scaling_factor(time)
                * velocity_difference
                * dt
            )

        return inter_vel_diff_score
