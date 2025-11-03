from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from laplace import Laplace

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.fiper_data_recorder.configuration_fiper_data_recorder import (
    FiperDataRecorderConfig,
)
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

from ..uncertainty_samplers.configuration_uncertainty_sampler import UncertaintySamplerConfig
from .ensemble_utils.factory import build_ensemble_models
from .laplace_utils.posterior_builder import get_laplace_posterior


@dataclass
class ScorerArtifacts:
    """
    Artifacts required by the uncertainty scorer.

    Attributes:
        ensemble_models: Model adapters used when scorer_type='ensemble'.
        laplace_posterior: Laplace posterior used when scorer_type='laplace'.
    """
    ensemble_models: List[BaseFlowMatchingAdapter] = None
    laplace_posterior: Optional[Laplace] = None

def build_scorer_artifacts_for_uncertainty_sampler(
    uncertainty_sampler_cfg: UncertaintySamplerConfig,
    policy: PreTrainedPolicy,
    dataset: LeRobotDataset,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    libero_tasks: dict[str, list[int]] | None = None,
) -> ScorerArtifacts:
    """
    Build scorer artifacts (ensemble model or Laplace posterior) from the active uncertainty sampler config.
    """
    active_config = uncertainty_sampler_cfg.active_config
    scorer_type = getattr(active_config, "scorer_type", None)
    if scorer_type is None:
        return ScorerArtifacts()
    if scorer_type == "ensemble":
        ensemble_models = build_ensemble_models(
            ensemble_model_paths=active_config.ensemble_model_paths,
            policy_cfg=policy.config,
            ds_meta=dataset.meta,
        )
        return ScorerArtifacts(ensemble_models=ensemble_models)
    if scorer_type == "laplace":
        laplace_posterior = get_laplace_posterior(
            policy=policy,
            preprocessor=preprocessor,
            dataset=dataset,
            laplace_config=active_config.laplace_config,
            libero_tasks=libero_tasks,
        )
        return ScorerArtifacts(laplace_posterior=laplace_posterior)
    raise ValueError(f"Unknown scorer_type: {scorer_type!r}")

def build_scorer_artifacts_for_fiper_recorder(
    fiper_data_recorder_cfg: FiperDataRecorderConfig,
    policy: PreTrainedPolicy,
    dataset: LeRobotDataset,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    libero_tasks: dict[str, list[int]] | None = None,
) -> ScorerArtifacts:
    """
    Build both ensemble model and Laplace posterior artifacts for the FIPER data recorder.
    """
    ensemble_models = build_ensemble_models(
        ensemble_model_paths=fiper_data_recorder_cfg.ensemble_model_paths,
        policy_cfg=policy.config,
        ds_meta=dataset.meta,
    )
    laplace_posterior = get_laplace_posterior(
        policy=policy,
        preprocessor=preprocessor,
        dataset=dataset,
        laplace_config=fiper_data_recorder_cfg.laplace_config,
        libero_tasks=libero_tasks,
    )
    return ScorerArtifacts(ensemble_models=ensemble_models, laplace_posterior=laplace_posterior)
