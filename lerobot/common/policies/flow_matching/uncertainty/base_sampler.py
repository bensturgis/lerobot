from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

from ..configuration_flow_matching import FlowMatchingConfig
from ..ode_solver import ADAPTIVE_SOLVERS, FIXED_STEP_SOLVERS, ODESolver


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
            num_action_seq_samples: How many action sequences and corresponding
                uncertainty scores to sample.
            extra_sampling_times: Extra times at which the sampling ODE should be evaluated.
        """
        self.flow_matching_cfg = flow_matching_cfg
        self.flow_matching_model = flow_matching_model
        self.velocity_model = flow_matching_model.unet
        self.sampling_ode_solver = ODESolver(self.velocity_model)
        self.num_action_seq_samples = num_action_seq_samples

        self.horizon = self.flow_matching_cfg.horizon
        self.action_dim = self.flow_matching_cfg.action_feature.shape[0]
        self.device = get_device_from_parameters(flow_matching_model)
        self.dtype = get_dtype_from_parameters(flow_matching_model)

        # Store latest sampled action sequences and their uncertainty scores for logging
        self.latest_action_candidates: Optional[Tensor] = None
        self.latest_uncertainties: Optional[Tensor] = None

        # Build time grid for sampling according to ODE solver method and scoring metric
        if flow_matching_cfg.ode_solver_method in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = self.sampling_ode_solver.make_sampling_time_grid(
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
    
    def _prepare_conditioning(self, global_cond: Tensor) -> Tensor:
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
    
    def compose_action_seqs(
        self,
        prev_action_seq: Tensor,
        new_action_seq: Tensor
    ) -> Tensor:
        """
        Stitch together a complete candidate action sequence by keeping the prefix that
        has already been executed and appending the freshly sampled suffix.

        Args:
            prev_action_seq: Sequence collected during the previous sampling step.
                Shape: (batch_size, horizon, action_dim).
            new_action_seq: Newly generated action sequence.
                Shape: (batch_size, horizon, action_dim).

        Returns:
            The composed action sequence. Shape: (batch_size, horizon, action_dim).
        """
        # Indices where to split and recompose the trajectory
        prev_action_seq_end = (
            self.flow_matching_cfg.n_obs_steps - 1 + self.flow_matching_cfg.n_action_steps
        )
        new_action_seqs_start = self.flow_matching_cfg.n_obs_steps - 1
        new_action_seqs_end = new_action_seqs_start + (self.horizon - prev_action_seq_end)
        
        # Repeat previous prefix to match batch dimension
        prev_action_sequence_duplicated = prev_action_seq.expand(
            self.num_action_seq_samples, -1, -1
        )
        
        # Compose full action sequences from stored prefix and newly sampled action sequences
        composed_action_seq = torch.cat([
            prev_action_sequence_duplicated[:, :prev_action_seq_end, :],
            new_action_seq[:, new_action_seqs_start:new_action_seqs_end, :]
        ], dim=1)

        return composed_action_seq

    @abstractmethod
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample num_action_seq_samples many action sequences and compute their
        uncertainty score according to some specific metric.

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.

        Returns:
            - Action sequences samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores. Shape: [num_action_seq_samples,]
        """
        pass