from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from lerobot.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
)

from .uncertainty_adapter import UncertaintyModelAdapter


class SmolVLAUncertaintyAdapter(UncertaintyModelAdapter):
    def __init__(self, config: SmolVLAConfig, model: VLAFlowMatching):
        super().__init__(model=model, config=config)

    @property
    def horizon(self) -> int:
        return self.config.chunk_size

    @property
    def action_dim(self) -> int:
        return self.config.max_action_dim

    @property
    def ode_solver_config(self) -> Dict[str, Any]:
        return {
            "solver_method": "euler",
            "step_size": 0.1,
            "atol": None,
            "rtol": None,
        }

    @property
    def cond_vf_config(self) -> Dict[str, Any]:
        return {
            "type": "ot",
            "sigma_min": 0,
            "beta_min": None,
            "beta_max": None,
        }

    @torch.no_grad()
    def prepare_conditioning(self, observation: dict[str, Tensor], num_action_samples: int) -> Dict[str, Tensor]:
        observation = self.expand_observation(observation=observation, num_action_samples=num_action_samples)

        images, img_masks = self.model.prepare_images(observation)
        state = self.model.prepare_state(observation)
        lang_tokens = observation[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = observation[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(pad_masks=prefix_pad_masks, att_masks=prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        return {
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    def make_velocity_fn(self, conditioning: Dict[str, Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def v_t(t: Tensor, x_t: Tensor) -> Tensor:
            return self.model.denoise_step(
                prefix_pad_masks=conditioning["prefix_pad_masks"],
                past_key_values=conditioning["past_key_values"],
                x_t=x_t,
                timestep=t.expand(x_t.shape[0]),
            )
        return v_t
