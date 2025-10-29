import copy
from typing import List

from lerobot import envs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.utils.utils import get_safe_torch_device


def build_ensemble_models(
    ensemble_model_paths: List[str],
    policy_cfg: PreTrainedConfig,
    env_cfg: envs.EnvConfig,
) -> List[BaseFlowMatchingAdapter]:
    """
    Load pretrained ensemble flow-matching models from the specified paths.
    """
    from lerobot.policies.factory import make_flow_matching_adapter_from_policy, make_policy

    device = get_safe_torch_device(policy_cfg.device)

    ensemble_adapters: List[BaseFlowMatchingAdapter] = []
    for ensemble_model_path in ensemble_model_paths:
        ensemble_model_config = copy.deepcopy(policy_cfg)
        ensemble_model_config.pretrained_path = ensemble_model_path
        ensemble_policy = make_policy(cfg=ensemble_model_config, env_cfg=env_cfg).to(device)
        ensemble_policy.eval()
        ensemble_adapters.append(
            make_flow_matching_adapter_from_policy(policy=ensemble_policy)
        )
    return ensemble_adapters
