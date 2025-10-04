from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
    make_sampling_time_grid,
)

from ..uncertainty_adapters.uncertainty_adapter import UncertaintyModelAdapter


class UncertaintySampler(ABC):
    """
    Abstract base class for uncertainty samplers that generates multiple action sequences and
    an aggregated uncertainty score from flow-matching-based models.
    """
    def __init__(
        self,
        model: UncertaintyModelAdapter,
        num_action_seq_samples: int,
        extra_sampling_times: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            model: A unified adapter that wraps a flow-matching model and exposes a common interface
                for the uncertainty sampler.
            num_action_seq_samples: How many action sequences to sample and use for uncertainty estimation.
            extra_sampling_times: Extra times at which the sampling ODE should be evaluated.
        """
        self.method_name = "base"
        self.policy_config = model.config
        self.model = model
        self.sampling_ode_solver = ODESolver()
        self.num_action_seq_samples = num_action_seq_samples

        self.horizon = model.horizon
        self.action_dim = model.action_dim
        self.device = model.device
        self.dtype = model.dtype

        # Store latest sampled action sequences and the uncertainty score for logging
        self.latest_action_candidates: Optional[Tensor] = None
        self.latest_uncertainty: Optional[float] = None

        # Build time grid for sampling according to ODE solver method and scoring metric
        if self.policy_config.ode_solver_method in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = make_sampling_time_grid(
                step_size=self.policy_config.ode_step_size,
                extra_times=extra_sampling_times,
                device=self.device
            )
        elif self.policy_config.ode_solver_method in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = torch.tensor(
                [0.0, *([] if extra_sampling_times is None else extra_sampling_times), 1.0],
                device=self.device, dtype=self.dtype
            )
        else:
            raise ValueError(f"Unknown ODE solver method: {self.policy_config.ode_solver_method}.")

    @abstractmethod
    def conditional_sample_with_uncertainty(
        self,
        conditioning: Dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, float]:
        """
        Sample num_action_seq_samples many action sequences and compute their aggregated
        uncertainty score according to some specific metric.

        Args:
            conditioning: Conditioning information for the velocity function derived from the current
                observation, language instruction, robot state, etc.
            generator: PyTorch random number generator.

        Returns:
            - Action sequences samples. Shape: (num_action_seq_samples, horizon, action_dim).
            - Uncertainty score.
        """
        raise NotImplementedError
