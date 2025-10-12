from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters


class BaseFlowMatchingAdapter(ABC):
    def __init__(self, model: nn.Module, config: PreTrainedConfig):
        self.model = model
        self.config = config

    @property
    @abstractmethod
    def horizon(self) -> int:
        raise NotImplementedError

    @property
    def n_action_steps(self) -> int:
        return self.config.n_action_steps

    @property
    def n_obs_steps(self) -> int:
        return self.config.n_obs_steps

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return get_device_from_parameters(self.model)

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype_from_parameters(self.model)

    @property
    @abstractmethod
    def ode_solver_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def cond_vf_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def expand_observation(self, observation: Dict[str, Tensor], num_action_samples: int) -> Dict[str, Tensor]:
        expanded_observation = {}
        for key, obs_tensor in observation.items():
            if not key.startswith("observation"):
                continue
            if obs_tensor.shape[0] == num_action_samples:
                expanded_observation[key] = obs_tensor
            elif obs_tensor.shape[0] == 1:
                expanded_observation[key] = obs_tensor.expand(num_action_samples, *obs_tensor.shape[1:])
            else:
                raise ValueError(
                    "Expected to prepare conditioning for a single observation (batch size=1), " \
                    f"but got {key} with batch_size={obs_tensor.shape[0]}."
                )

        return expanded_observation

    @abstractmethod
    def prepare_conditioning(self, observation: dict[str, Tensor], num_action_samples: int) -> Dict[str, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def sample_prior(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        return torch.randn(
            (num_samples, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
            generator=generator
        )

    @abstractmethod
    def make_velocity_fn(self, conditioning: Dict[str, Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        raise NotImplementedError
