from torch import nn

from lerobot.configs.policies import PreTrainedConfig

from .uncertainty_adapter import UncertaintyModelAdapter


def make_uncertainty_adapter(model: nn.Module, policy_config: PreTrainedConfig) -> UncertaintyModelAdapter:
    if policy_config.type == "flow_matching":
        from lerobot.uncertainty.uncertainty_adapters.flow_matching_uncertainty_adapter import (
            FlowMatchingUncertaintyAdapter,
        )

        return FlowMatchingUncertaintyAdapter(config=policy_config, model=model)
    elif policy_config.type == "smolvla":
        from lerobot.uncertainty.uncertainty_adapters.smolvla_uncertainty_adapter import (
            SmolVLAUncertaintyAdapter,
        )

        return SmolVLAUncertaintyAdapter(config=policy_config, model=model)
    else:
        raise ValueError(f"No uncertainty adapter available for policy type '{policy_config.type}'.")
