from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import Tensor

from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingModel

from .uncertainty_adapter import UncertaintyModelAdapter


class FlowMatchingUncertaintyAdapter(UncertaintyModelAdapter):
    def __init__(self, config: FlowMatchingConfig, model: FlowMatchingModel):
        super().__init__(model=model, config=config)

    @property
    def horizon(self) -> int:
        return self.config.horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_feature.shape[0]

    @torch.no_grad()
    def prepare_conditioning(self, observation: Dict[str, Tensor]) -> Dict[str, Tensor]:
        global_cond = self.model.prepare_global_conditioning(observation)

        if global_cond.ndim == 1:
            global_cond = global_cond.unsqueeze(0)
        if global_cond.ndim != 2 or global_cond.size(0) != 1:
            raise ValueError(
                f"Expected `global_cond` to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )

        return {
            "global_cond": global_cond,
        }

    @torch.no_grad()
    def sample_prior(
        self,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        return torch.randn(
            num_samples, self.horizon, self.action_dim, device=device, dtype=dtype, generator=generator
        )

    def make_velocity_fn(self, conditioning: Dict[str, Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def v_t(t: Tensor, x_t: Tensor) -> Tensor:
            batch_size = x_t.shape[0]
            return self.model.unet(
                x_t, t.expand(batch_size), global_cond=conditioning["global_cond"].expand(batch_size, -1)
            )
        return v_t
