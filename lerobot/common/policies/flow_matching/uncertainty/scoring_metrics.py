from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor
from torch.distributions import Independent, Normal

from lerobot.common.policies.flow_matching.conditional_probability_path import (
    OTCondProbPath,
    VPDiffusionCondProbPath,
)
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingConditionalUnet1d
from lerobot.common.policies.flow_matching.ode_solver import ADAPTIVE_SOLVERS, FIXED_STEP_SOLVERS, ODESolver
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

from .base_sampler import FlowMatchingUncertaintySampler
from .configuration_uncertainty_sampler import ScoringMetricConfig


class FlowMatchingUncertaintyMetric(ABC):  # noqa: B024
    """Abstract base class for flow matching uncertainty metrics."""
    name: str = "base"
    type: str = "base"


class TerminalStateMetric(FlowMatchingUncertaintyMetric, ABC):
    """
    Abstract base class for uncertainty metrics that operate on terminal action sequences,
    i.e., the final outputs of the flow matching ODE integration. 
    Subclasses define how the velocity model is applied to score these sequences 
    and produce an uncertainty score.
    """
    type: str = "terminal"

    @abstractmethod
    def __call__(
        self,
        velocity_model: FlowMatchingConditionalUnet1d,
        global_cond: Tensor,
        action_sequences: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute an uncertainty score for a batch of action sequences using a scorer
        flow matching model.
        Args:
            velocity_model: Flow matching velocity model for scoring.
            scorer_global_cond: Conditioning vector for the scorer velocity model.
                Shape: (batch_size, cond_dim).
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
        velocity_model: FlowMatchingConditionalUnet1d,
        global_cond: Tensor,
        action_sequences: Tensor,
        **_: Any
    ) -> Tensor:
        """
        Evaluate the velocity only at the terminal action sequences for multiple evaluation
        times and return the mean L2 norm as the uncertainty score.
        """
        device = get_device_from_parameters(velocity_model)
        dtype = get_dtype_from_parameters(velocity_model)
        # Evaluate velocity on the final sampled sequence
        terminal_vel_norms: list[float] = []
        for time in self.velocity_eval_times:
            time_batch = torch.full(
                (action_sequences.shape[0],), time, device=device, dtype=dtype
            )
            velocity = velocity_model(
                action_sequences,
                time_batch,
                global_cond,
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
        """
        self.velocity_eval_times = config.velocity_eval_times

    def __call__(
        self,
        velocity_model: FlowMatchingConditionalUnet1d,
        global_cond: Tensor,
        action_sequences: Tensor,
        **_: Any,
    ) -> Tensor:
        """
        Compute the proxy "distance-from-mode" score computed by averaging (1 - t) * ‖v(x; t)‖
        of the velocity at the terminal action sequence x across the specified 
        evaluation times.
        """
        device = get_device_from_parameters(velocity_model)
        dtype = get_dtype_from_parameters(velocity_model)
        distances: list[float] = []
        # Loop over each time in [0, 1) at which we want to probe the velocity field
        for time in self.velocity_eval_times:
            time_batch = torch.full(
                (action_sequences.shape[0],), time, device=device, dtype=dtype
            )
            # Query the velocity field at the terminal action sequence and this time
            velocity = velocity_model(
                action_sequences,
                time_batch,
                global_cond,
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

    def __init__(self, config: ScoringMetricConfig, uncertainty_sampler: FlowMatchingUncertaintySampler):           
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
        self.lik_estimation_time_grid = self._get_lik_estimation_time_grid()        

    def _get_lik_estimation_time_grid(self) -> Tensor:
        """
        Build time grid to estimate likelihood according to ODE solver method.

        For a fixed step solver the time grid consists of a fine segment of 10 points evenly
        spaced from 1.0 up to 0.93 and a coarse segment of 10 points evenly spaced from 0.9 up to 0.0.

        Returns:
            A 1D time grid.
        """
        if self.lik_ode_solver_cfg.method in FIXED_STEP_SOLVERS:
            fine = torch.linspace(1.0, 0.93, steps=10, device=self.device, dtype=self.dtype)
            coarse = torch.linspace(0.9, 0.0,  steps=10, device=self.device, dtype=self.dtype)
            return torch.cat((fine, coarse))
        elif self.lik_ode_solver_cfg.method in ADAPTIVE_SOLVERS:
            lik_estimation_time_grid = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)
        else:
            raise ValueError(
                f"Unknown ODE solver method {self.lik_ode_solver_cfg.method}. "
                f"Expected one of {sorted(FIXED_STEP_SOLVERS | ADAPTIVE_SOLVERS)}."
            )

        return lik_estimation_time_grid

    def __call__(
        self,
        velocity_model: FlowMatchingConditionalUnet1d,
        global_cond: Tensor,
        action_sequences: Tensor,
        generator: Optional[torch.Generator] = None,
        **_: Any,
    ) -> Tensor:
        """
        Run a reverse-time ODE under the scorer model to compute the log-likelihood of the 
        action sequence; the score is the negative log-likelihood.
        """
        # Compute log-likelihood of sampled action sequences in scorer model    
        scoring_ode_solver = ODESolver(velocity_model)
        _, log_probs = scoring_ode_solver.sample_with_log_likelihood(
            x_init=action_sequences,
            time_grid=self.lik_estimation_time_grid,
            global_cond=global_cond,
            log_p_0=self.gaussian_log_density,
            method=self.lik_ode_solver_cfg.method,
            atol=self.lik_ode_solver_cfg.atol,
            rtol=self.lik_ode_solver_cfg.rtol,
            exact_divergence=self.lik_ode_solver_cfg.exact_divergence,
            generator=generator,
        )

        # Use negative log-likelihood as uncertainty score
        return -log_probs
    

class InterVelDiff(FlowMatchingUncertaintyMetric):
    """
    Uncertainty metric based on intermediate velocity discrepancies.
    
    Compares the velocities predicted by two flow matching models along their 
    respective ODE trajectories. Large values indicate stronger disagreement 
    between the reference and comparison models.
    """
    name: str = "inter_vel_diff"
    type: str = "trajectory"

    def __init__(self, config: ScoringMetricConfig, uncertainty_sampler: FlowMatchingUncertaintySampler):
        """
        Args:
            config: Scoring metric settings.
        """
        self.velocity_eval_times = config.velocity_eval_times
        self.device = uncertainty_sampler.device
        self.dtype = uncertainty_sampler.dtype
        self.ode_solver = uncertainty_sampler.sampling_ode_solver
        self.sampling_time_grid = uncertainty_sampler.sampling_time_grid
        self.cond_vf_type = uncertainty_sampler.flow_matching_cfg.cond_vf_type
        if self.cond_vf_type == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=uncertainty_sampler.flow_matching_cfg.beta_min,
                beta_max=uncertainty_sampler.flow_matching_cfg.beta_max,
            )
        else:
            self.cond_prob_path = OTCondProbPath()
    
    def _get_scaling_factor(self, t: Tensor) -> Tensor:
        """
        Compute the time-dependent scaling factor used to weight velocity differences 
        based on conditional vector field type.
        """
        if self.cond_vf_type == "vp":
            return (2 / self.cond_prob_path.get_beta(t))
        elif self.cond_vf_type == "ot":
            return t
        else:
            raise ValueError(
                "No intermediate velocity difference factor provided for conditional " \
                f"VF type: {self.cond_vf_type}."
            )

    def __call__(
        self,
        ref_ode_states: Tensor,
        ref_velocity_model: FlowMatchingConditionalUnet1d,
        ref_global_cond: Tensor,
        cmp_ode_states: Tensor,
        cmp_velocity_model: FlowMatchingConditionalUnet1d,
        cmp_global_cond: Tensor,
    ) -> Tensor:
        """
        Compute the average velocity discrepancy between two trajectories at intermediate ODE states.

        For each evaluation time t, this metric computes the L2 difference between the velocity 
        predicted by the reference model at `x_ref(t)` and the comparison model at `x_cmp(t)`:

            ||v_ref(x_ref(t), t) - v_cmp(x_cmp(t), t)||

        The discrepancies are integrated over time and averaged across the evaluation times.

        Args:
            ref_ode_states: ODE integration states of the reference trajectory.
                Shape: (timesteps, batch_size, horizon, action_dim).
            ref_velocity_model: Velocity model associated with the reference trajectory.
            ref_global_cond: Conditioning vector for the reference model.
                Shape: (batch_size, cond_dim).
            cmp_ode_states: ODE integration states of the comparison trajectory.
            cmp_velocity_model: Velocity model associated with the comparison trajectory.
            cmp_global_cond: Conditioning vector for the comparison model.

        Returns:
            Uncertainty scores per trajectory sample, where larger values indicate stronger
            disagreement between reference and comparison velocities. Shape: (batch_size,).
        """
        # Select the ODE states that correspond to the velocity evaluation times
        selected_ref_ode_states, selected_ref_grid_times = self.ode_solver.select_ode_states(
            time_grid=self.sampling_time_grid,
            ode_states=ref_ode_states,
            requested_times=torch.tensor(self.velocity_eval_times, device=self.device, dtype=self.dtype)
        )
        selected_cmp_ode_states, selected_cmp_grid_times = self.ode_solver.select_ode_states(
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
        batch_size = ref_ode_states[-1].shape[0]
        inter_vel_diff_score: Tensor = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        for idx, (time, ref_inter_state, cmp_inter_state) in enumerate(zip(
            selected_ref_grid_times, selected_ref_ode_states, selected_cmp_ode_states, strict=False
        )):
            # Determine dt: difference to next time or to 1.0 for last step
            dt = selected_ref_grid_times[idx + 1] - time if idx < len(selected_ref_grid_times) - 1 else 1.0 - time
        
            time_batch = torch.full(
                (batch_size,), time, device=self.device, dtype=self.dtype
            )
            ref_velocity = ref_velocity_model(
                ref_inter_state,
                time_batch,
                ref_global_cond,
            )
            cmp_velocity = cmp_velocity_model(
                cmp_inter_state,
                time_batch,
                cmp_global_cond,
            )
            # L2 norm across time and action dims gives magnitude of velocity difference
            velocity_difference = torch.norm(ref_velocity - cmp_velocity, dim=(1, 2)) ** 2
            
            # Scale velocity difference by factor that depends on conditional vector field type
            inter_vel_diff_score += (
                self._get_scaling_factor(time)
                * velocity_difference
                * dt
            )
        
        return inter_vel_diff_score