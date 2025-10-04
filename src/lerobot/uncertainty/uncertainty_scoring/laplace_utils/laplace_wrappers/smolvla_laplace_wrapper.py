from typing import Callable

from torch import Tensor

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

from .laplace_wrapper import LaplaceBatch, LaplaceWrapper


class SmolVLALaplaceBatch(LaplaceBatch):
    def to(self, *args, **kwargs) -> "SmolVLALaplaceBatch":
        raise NotImplementedError("Laplace approximation not availabe for SmolVLA.")

    def detach(self) -> "SmolVLALaplaceBatch":
        raise NotImplementedError("Laplace approximation not availabe for SmolVLA.")

    def cpu(self) -> "SmolVLALaplaceBatch":
        raise NotImplementedError("Laplace approximation not availabe for SmolVLA.")


class SmolVLALaplaceWrapper(LaplaceWrapper):
    def __init__(self, config: SmolVLAConfig, model: VLAFlowMatching):
        super().__init__(model=model, config=config)

    def build_collate_fn(self) -> Callable:
        raise NotImplementedError("Laplace approximation not availabe for SmolVLA.")

    def forward(self, batch: LaplaceBatch) -> Tensor:
        raise NotImplementedError("Laplace approximation not availabe for SmolVLA.")

    def apply_laplace_scope(self, scope: str):
        raise NotImplementedError("Laplace approximation not availabe for SmolVLA.")
