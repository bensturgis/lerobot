from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from laplace import Laplace

from lerobot.common import envs
from lerobot.common.policies.flow_matching.modelling_flow_matching import (
    FlowMatchingModel,
    FlowMatchingPolicy,
)
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig

from .configuration_uncertainty_sampler import UncertaintySamplerConfig
from .laplace_utils import (
    create_laplace_flow_matching_calib_loader,
    get_laplace_posterior,
    make_laplace_path,
)


@dataclass
class ScorerArtifacts:
    """
    Artifacts required by the uncertainty scorer.

    Attributes:
        ensemble_model: Flow-matching model used when scorer_type='ensemble'.
        laplace_posterior: Laplace posterior used when scorer_type='laplace'.
    """
    ensemble_model: Optional[FlowMatchingModel] = None
    laplace_posterior: Optional[Laplace] = None

def build_scorer_artifacts(
    uncertainty_sampler_cfg: UncertaintySamplerConfig,
    policy_cfg: PreTrainedConfig,
    env_cfg: envs.EnvConfig,
    dataset_cfg: DatasetConfig,
    policy: FlowMatchingPolicy,
) -> ScorerArtifacts:
    """
    Build scorer artifacts from the *top-level* sampler config.

    Uses the active sub-config. If it has no `scorer_type` (e.g., composed_sequence, entropy),
    returns empty artifacts. Otherwise constructs either an ensemble flow-matching model
    or a (loaded/fitted) Laplace posterior.

    Args:
        uncertainty_sampler_cfg: Top-level uncertainty sampler config.
        policy_cfg: Policy config (for device and pretrained path).
        env_cfg: Environment config (used to instantiate an ensemble policy).
        dataset_cfg: Dataset config (for Laplace path/loader).
        policy: Main policy; its `.flow_matching` is used for Laplace.

    Returns:
        ScorerArtifacts with exactly one of `ensemble_model` or `laplace_posterior` set,
        or both None if the active config has no scorer.
    """
    device = get_safe_torch_device(policy_cfg.device)
    active_config = uncertainty_sampler_cfg.active_config
    scorer_type = getattr(active_config, "scorer_type", None)

    # No scorer
    if scorer_type is None:
        return ScorerArtifacts()

    if scorer_type == "ensemble":
        from lerobot.common.policies.factory import make_policy
        
        ensemble_path = getattr(active_config, "ensemble_model_path", None)
        if not ensemble_path:
            raise ValueError("For scorer_type='ensemble', ensemble_model_path must be provided.")
        ensemble_policy_cfg = copy.deepcopy(policy_cfg)
        ensemble_policy_cfg.pretrained_path = ensemble_path
        ensemble_policy = make_policy(
            cfg=ensemble_policy_cfg,
            env_cfg=env_cfg,
        ).to(device)
        ensemble_policy.eval().requires_grad_(False)
        return ScorerArtifacts(ensemble_model=ensemble_policy.flow_matching)

    if scorer_type == "laplace":
        laplace_path: Path = make_laplace_path(
            repo_id=dataset_cfg.repo_id,
            scope=active_config.laplace_scope,
            calib_fraction=active_config.calib_fraction,
        )
        calib_loader = None
        if not laplace_path.exists():
            calib_loader = create_laplace_flow_matching_calib_loader(
                dataset_cfg=dataset_cfg,
                policy_cfg=policy_cfg,
                policy=policy,
                calib_fraction=active_config.calib_fraction,
                batch_size=active_config.batch_size,
            )
        laplace_posterior = get_laplace_posterior(
            cfg=active_config,
            flow_matching_model=policy.flow_matching,
            laplace_calib_loader=calib_loader,
            laplace_path=laplace_path,
        )
        return ScorerArtifacts(laplace_posterior=laplace_posterior)

    raise ValueError(f"Unknown scorer_type: {scorer_type!r}")