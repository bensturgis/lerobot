from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from laplace import Laplace

from lerobot.common import envs
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig

from ...fiper_data_recording.configuration_fiper_data_recorder import (
    FiperDataRecorderConfig,
)
from ...modelling_flow_matching import (
    FlowMatchingModel,
    FlowMatchingPolicy,
)
from ..configuration_uncertainty_sampler import UncertaintySamplerConfig
from .ensemble_utils import build_ensemble_model
from .laplace_utils import build_laplace_posterior_artifact


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

def build_scorer_artifacts_for_uncertainty_sampler(
    uncertainty_sampler_cfg: UncertaintySamplerConfig,
    policy_cfg: PreTrainedConfig,
    env_cfg: envs.EnvConfig,
    dataset_cfg: DatasetConfig,
    policy: FlowMatchingPolicy,
) -> ScorerArtifacts:
    """
    Build scorer artifacts (ensemble model or Laplace posterior) 
    from the active uncertainty sampler config.
    """
    active_config = uncertainty_sampler_cfg.active_config
    scorer_type = getattr(active_config, "scorer_type", None)
    if scorer_type is None:
        return ScorerArtifacts()
    if scorer_type == "ensemble":
        ensemble_model = build_ensemble_model(
            ensemble_model_path=active_config.ensemble_model_path,
            policy_cfg=policy_cfg,
            env_cfg=env_cfg,
        )
        return ScorerArtifacts(ensemble_model=ensemble_model)
    if scorer_type == "laplace":
        post = build_laplace_posterior_artifact(
            laplace_scope=active_config.laplace_scope,
            calib_fraction=active_config.calib_fraction,
            batch_size=active_config.batch_size,
            dataset_cfg=dataset_cfg,
            policy_cfg=policy_cfg,
            policy=policy
        )
        return ScorerArtifacts(laplace_posterior=post)
    raise ValueError(f"Unknown scorer_type: {scorer_type!r}")

def build_scorer_artifacts_for_fiper_recorder(
    fiper_data_recorder_cfg: FiperDataRecorderConfig,
    policy_cfg: PreTrainedConfig,
    env_cfg: envs.EnvConfig,
    dataset_cfg: DatasetConfig,
    policy: FlowMatchingPolicy,
) -> ScorerArtifacts:
    """
    Build both ensemble model and Laplace posterior artifacts 
    for the FIPER data recorder.
    """
    ensemble_model = build_ensemble_model(
        ensemble_model_path=fiper_data_recorder_cfg.ensemble_model_path,
        policy_cfg=policy_cfg,
        env_cfg=env_cfg,
    )
    laplace_posterior = build_laplace_posterior_artifact(
        laplace_scope=fiper_data_recorder_cfg.laplace_scope,
        calib_fraction=fiper_data_recorder_cfg.calib_fraction,
        batch_size=fiper_data_recorder_cfg.batch_size,
        dataset_cfg=dataset_cfg,
        policy_cfg=policy_cfg,
        policy=policy,
    )
    return ScorerArtifacts(ensemble_model=ensemble_model, laplace_posterior=laplace_posterior)
