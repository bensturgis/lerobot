from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import PolicyProcessorPipeline


@dataclass
class LaplaceBatch:
    @abstractmethod
    def to(self, *args, **kwargs) -> "LaplaceBatch":
        raise NotImplementedError

    @abstractmethod
    def detach(self) -> "LaplaceBatch":
        raise NotImplementedError

    @abstractmethod
    def cpu(self) -> "LaplaceBatch":
        raise NotImplementedError

class LaplaceWrapper(nn.Module):
    def __init__(self, model: nn.Module, config: PreTrainedConfig):
        super().__init__()
        self.model = model
        self.config = config

    @property
    def device(self) -> int:
        return self.config.device

    @abstractmethod
    def build_collate_fn(
        self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    ) -> Callable[[Dict[str, Tensor]], Tuple[LaplaceBatch, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch: LaplaceBatch) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply_laplace_scope(self, scope: str):
        raise NotImplementedError
