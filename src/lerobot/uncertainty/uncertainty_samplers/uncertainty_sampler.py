from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.common.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
    make_sampling_time_grid,
)


class UncertaintySampler(ABC):
    """
    Abstract base class for uncertainty samplers that generates multiple action sequences and
    an aggregated uncertainty score from flow-matching-based models.
    """
    def __init__(
        self,
        model: BaseFlowMatchingAdapter,
        num_action_samples: int,
        extra_sampling_times: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            model: A unified flow matching adapter that wraps a flow-matching model and exposes a common
                interface for the uncertainty sampler.
            num_action_samples: How many action sequences to sample and use for uncertainty estimation.
            extra_sampling_times: Extra times at which the sampling ODE should be evaluated.
        """
        self.method_name = "base"
        self.model = model
        self.sampling_ode_solver = ODESolver()
        self.num_action_samples = num_action_samples

        self.horizon = model.horizon
        self.n_action_steps = model.n_action_steps
        self.n_obs_steps = model.n_obs_steps
        self.action_dim = model.action_dim
        self.device = model.device
        self.dtype = model.dtype
        self.cond_vf_config = model.cond_vf_config

        # Store latest sampled action sequences and the uncertainty score for logging
        self.latest_uncertainty: Optional[float] = None

        # Build time grid for sampling according to ODE solver method and scoring metric
        self.ode_solver_config = model.ode_solver_config
        if self.ode_solver_config["solver_method"] in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = make_sampling_time_grid(
                step_size=self.ode_solver_config["step_size"],
                extra_times=extra_sampling_times,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.ode_solver_config["solver_method"] in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = torch.tensor(
                [0.0, *([] if extra_sampling_times is None else extra_sampling_times), 1.0],
                device=self.device, dtype=self.dtype
            )
        else:
            raise ValueError(f"Unknown ODE solver method: {self.ode_solver_config['solver_method']}.")

    def rand_pick_action(self, action_candidates: Tensor) -> Tuple[Tensor, int]:
        """
        Randomly select one action sequence from a batch.

        Args:
            action_candidates: Batch candidate action sequences. Shape: (num_action_samples, horizon, action_dim).

        Returns:
            - Selected action sequence. Shape: (1, horizon, action_dim)
            - Index of the selected action sequence.
        """
        rand_idx = torch.randint(
            low=0,
            high=self.num_action_samples,
            size=(1,),
            device=action_candidates.device
        ).item()
        actions = action_candidates[rand_idx : rand_idx+1]

        return actions.to(device="cpu", dtype=torch.float32), rand_idx

    @abstractmethod
    def conditional_sample_with_uncertainty(
        self,
        conditioning: Dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, float]:
        """
        Sample num_action_samples many action sequences and compute their aggregated
        uncertainty score according to some specific metric.

        Args:
            conditioning: Conditioning information for the velocity function derived from the current
                observation, language instruction, robot state, etc.
            generator: PyTorch random number generator.

        Returns:
            - Action sequences samples. Shape: (num_action_samples, horizon, action_dim).
            - Uncertainty score.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        raise NotImplementedError
