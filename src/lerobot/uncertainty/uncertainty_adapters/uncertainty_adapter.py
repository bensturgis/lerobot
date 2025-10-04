from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters


class UncertaintyModelAdapter(ABC):
    def __init__(self, model: nn.Module, config: PreTrainedConfig):
        self.model = model
        self.config = config

    @property
    @abstractmethod
    def horizon(self) -> int:
        raise NotImplementedError

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

    @abstractmethod
    def prepare_conditioning(self, observation: dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def sample_prior(
        self,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def make_velocity_fn(self, conditioning: Dict[str, Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        raise NotImplementedError
