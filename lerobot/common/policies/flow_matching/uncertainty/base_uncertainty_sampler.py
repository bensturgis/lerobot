from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

from ..configuration_flow_matching import FlowMatchingConfig
from ..ode_solver import ADAPTIVE_SOLVERS, FIXED_STEP_SOLVERS, ODESolver, make_sampling_time_grid


class FlowMatchingUncertaintySampler(ABC):
    """
    Abstract base class for uncertainty samplers that sample multiple action sequences
    and their per-sample uncertainty based on a Flow Matching model.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        flow_matching_model: FlowMatchingModel,
        num_action_seq_samples: int,
        extra_sampling_times: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            flow_matching_cfg: Shared configuration object for Flow Matching settings.
            flow_matching_model: The learned flow matching model.
            num_action_seq_samples: How many action sequences to sample and use for uncertainty
                estimation.
            extra_sampling_times: Extra times at which the sampling ODE should be evaluated.
        """
        self.method_name = "base"
        self.flow_matching_cfg = flow_matching_cfg
        self.flow_matching_model = flow_matching_model
        self.velocity_model = flow_matching_model.unet
        self.sampling_ode_solver = ODESolver(self.velocity_model)
        self.num_action_seq_samples = num_action_seq_samples

        self.horizon = self.flow_matching_cfg.horizon
        self.action_dim = self.flow_matching_cfg.action_feature.shape[0]
        self.device = get_device_from_parameters(flow_matching_model)
        self.dtype = get_dtype_from_parameters(flow_matching_model)

        # Store latest sampled action sequences and the uncertainty score for logging
        self.latest_action_candidates: Optional[Tensor] = None
        self.latest_uncertainty: Optional[float] = None

        # Build time grid for sampling according to ODE solver method and scoring metric
        if flow_matching_cfg.ode_solver_method in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = make_sampling_time_grid(
                step_size=flow_matching_cfg.ode_step_size,
                extra_times=extra_sampling_times,
                device=self.device
            )
        elif flow_matching_cfg.ode_solver_method in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = torch.tensor(
                [0.0, *([] if extra_sampling_times is None else extra_sampling_times), 1.0],
                device=self.device, dtype=self.dtype
            )
        else:
            raise ValueError(f"Unknown ODE solver method: {flow_matching_cfg.ode_solver_method}.")
        
        # Indices marking the portion of the trajectory that will actually be executed
        self.exec_start_idx = self.flow_matching_cfg.n_obs_steps - 1
        self.exec_end_idx = self.exec_start_idx + self.flow_matching_cfg.n_action_steps
    
    def _reshape_conditioning(self, global_cond: Tensor) -> Tensor:
        """
        Reshape single global conditioning vector to (num_action_seq_samples, cond_dim).
        """
        if global_cond.ndim == 1:
            global_cond = global_cond.unsqueeze(0)
        if global_cond.ndim != 2 or global_cond.size(0) != 1:
            raise ValueError(
                f"Expected `global_cond` to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )
        return global_cond.repeat(self.num_action_seq_samples, 1)
    
    def compose_ode_states(
        self,
        prev_ode_states: Tensor,
        new_ode_states: Tensor
    ) -> Tensor:
        """
        Splice ODE states by keeping the executed prefix from the previous rollout and appending the freshly
        sampled suffix from the new action generation. Inputs can be full ODE integration states with time
        dimension or final sampled action sequences only.

        Args:
            prev_action_seq: ODE states collected during the previous action generation step.
                Shape: (timesteps, 1, horizon, action_dim) or (1, horizon, action_dim) for final action sequences.
            new_action_seq: Newly generated action sequence or ODE states.
                Shape: (timesteps, batch_size, horizon, action_dim) or (batch_size, horizon, action_dim) for
                final action sequences.

        Returns:
            The composed ODE states. Shape: (timesteps, batch_size, horizon, action_dim) or
                (batch_size, horizon, action_dim) for final action sequences.
        """       
        def add_time_dimension(ode_states: Tensor) -> tuple[Tensor, bool]:
            if ode_states.ndim == 3:   # (batch_size, horizon, action_dim)
                return ode_states.unsqueeze(0), False  # (1, batch_size, horizon, action_dim)
            if ode_states.ndim == 4:   # (time_step, batch_size, horizon, action_dim)
                return ode_states, True
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(ode_states.shape)}")
        
        prev_ode_states, prev_had_time_dim = add_time_dimension(prev_ode_states)
        new_ode_states, new_had_time_dim = add_time_dimension(new_ode_states)

        if prev_ode_states.shape[1] != 1:
            raise ValueError(
                "Selected ODE states from previous action generation are expected to have batch size "
                f"of one but got batch_size={prev_ode_states.shape[1]}."
            )

        if [prev_ode_states.size(i) for i in (0, 2, 3)] != [new_ode_states.size(i) for i in (0, 2, 3)]:
            raise ValueError(
                "ODE states to compose are expected to have the same time dimension, horizon, "
                f"and action dimension. Got shapes {prev_ode_states.shape} and {new_ode_states.shape}."
            )

        new_action_seq_end = self.exec_start_idx + (self.horizon - self.exec_end_idx)
        
        # Repeat prefix from previous ODE states to match batch dimension
        prev_ode_states_duplicated = prev_ode_states.expand(
            -1, self.num_action_seq_samples, -1, -1
        )
        
        # Compose from stored prefix and newly generated ODE states
        composed_ode_states = torch.cat([
            prev_ode_states_duplicated[:, :, :self.exec_end_idx, :],
            new_ode_states[:, :, self.exec_start_idx:new_action_seq_end, :]
        ], dim=2)

        if not (prev_had_time_dim and new_had_time_dim):
            return composed_ode_states.squeeze(0)
        else:
            return composed_ode_states

    @abstractmethod
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, float]:
        """
        Sample num_action_seq_samples many action sequences and compute their
        uncertainty score according to some specific metric.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape: [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.

        Returns:
            - Action sequences samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty score.
        """
        raise NotImplementedError