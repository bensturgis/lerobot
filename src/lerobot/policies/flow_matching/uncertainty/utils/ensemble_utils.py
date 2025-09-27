import copy
from pathlib import Path
from typing import Union

from lerobot import envs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.flow_matching.modelling_flow_matching import (
    FlowMatchingModel,
)
from lerobot.utils.utils import get_safe_torch_device


def build_ensemble_model(
    ensemble_model_path: Union[Path, str],
    policy_cfg: PreTrainedConfig,
    env_cfg: envs.EnvConfig,
) -> FlowMatchingModel:
    """
    Load a pretrained ensemble flow-matching model from the given path.
    """
    from lerobot.policies.factory import make_policy

    device = get_safe_torch_device(policy_cfg.device)
    ensemble_model_config = copy.deepcopy(policy_cfg)
    ensemble_model_config.pretrained_path = str(ensemble_model_path)
    ensemble_policy = make_policy(cfg=ensemble_model_config, env_cfg=env_cfg).to(device)
    ensemble_policy.eval()
    return ensemble_policy.flow_matching
