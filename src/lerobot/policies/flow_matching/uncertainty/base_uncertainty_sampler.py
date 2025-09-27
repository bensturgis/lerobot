from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters

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
