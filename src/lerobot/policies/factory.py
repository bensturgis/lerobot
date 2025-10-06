#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Any, TypedDict

import torch
from torch import nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.uncertainty.uncertainty_adapters.factory import make_uncertainty_adapter
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    ScoringMetricConfig,
    UncertaintySamplerConfig,
)
from lerobot.uncertainty.uncertainty_samplers.uncertainty_sampler import (
    UncertaintySampler,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import (
    ScorerArtifacts,
)
from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import UncertaintyMetric


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    This function uses dynamic imports to avoid loading all policy classes into memory
    at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the policy. Supported names are "tdmpc", "diffusion", "flow_matching",
            "act", "vqbet", "pi0", "pi0fast", "sac", "reward_classifier", "smolvla".

    Returns:
        The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy name is not recognized.
    """
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "flow_matching":
        from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy

        return FlowMatchingPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy

        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        return Classifier
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The type of the policy. Supported types include "tdmpc", "diffusion",
            "flow_matching", "act", "vqbet", "pi0", "pi0fast", "sac", "smolvla", "reward_classifier".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "flow_matching":
        return FlowMatchingConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    if pretrained_path:
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # Create a new processor based on policy type
    if isinstance(policy_cfg, TDMPCConfig):
        from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_pre_post_processors

        processors = make_tdmpc_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, DiffusionConfig):
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

        processors = make_diffusion_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, FlowMatchingConfig):
        from lerobot.policies.flow_matching.processor_flow_matching import (
            make_flow_matching_pre_post_processors,
        )

        processors = make_flow_matching_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, ACTConfig):
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        processors = make_act_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, VQBeTConfig):
        from lerobot.policies.vqbet.processor_vqbet import make_vqbet_pre_post_processors

        processors = make_vqbet_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI0Config):
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        processors = make_pi0_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI0FASTConfig):
        from lerobot.policies.pi0fast.processor_pi0fast import make_pi0fast_pre_post_processors

        processors = make_pi0fast_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SACConfig):
        from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors

        processors = make_sac_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, RewardClassifierConfig):
        from lerobot.policies.sac.reward_model.processor_classifier import make_classifier_processor

        processors = make_classifier_processor(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SmolVLAConfig):
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        processors = make_smolvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        raise NotImplementedError(f"Processor for policy type '{policy_cfg.type}' is not implemented.")

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.

    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {}
    cfg.input_features = {}
    for key, ft in features.items():
        converted_key = key.replace("_", ".")
        if ft.type is FeatureType.ACTION:
            cfg.output_features[converted_key] = ft
        else:
            cfg.input_features[converted_key] = ft
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy


def make_uncertainty_sampler(
    uncertainty_sampler_config: UncertaintySamplerConfig,
    policy_config: PreTrainedConfig,
    model: nn.Module,
    scorer_artifacts: ScorerArtifacts,
) -> UncertaintySampler:
    # Initialize the uncertainty adapter
    uncertainty_adapter = make_uncertainty_adapter(
        model=model,
        policy_config=policy_config
    )
    
    if uncertainty_sampler_config.type == "composed_cross_bayesian":
        from lerobot.uncertainty.uncertainty_samplers.composed_cross_bayesian_sampler import (
            ComposedCrossBayesianSampler,
        )

        if uncertainty_sampler_config.composed_cross_bayesian_sampler.scorer_type == "ensemble" and scorer_artifacts.ensemble_adapter is None:
            raise ValueError(
                "Composed Cross-Bayesian uncertainty sampler with scorer_type=ensemble requires an ensemble model."
            )
        if uncertainty_sampler_config.composed_cross_bayesian_sampler.scorer_type == "laplace" and scorer_artifacts.laplace_posterior is None:
            raise ValueError(
                "Composed Cross-Bayesian uncertainty sampler with scorer_type=laplace requires Laplace posterior "
                "to draw a scorer model from."
            )

        return ComposedCrossBayesianSampler(
            flow_matching_cfg=flow_matching_cfg,
            cfg=uncertainty_sampler_config.composed_cross_bayesian_sampler,
            sampler_model=flow_matching_model,
            scorer_artifacts=scorer_artifacts,
        )
    if uncertainty_sampler_config.type == "composed_sequence":
        from lerobot.uncertainty.uncertainty_samplers.composed_seq_sampler import (
            ComposedSequenceSampler,
        )

        return ComposedSequenceSampler(
            config=uncertainty_sampler_config.composed_sequence_sampler,
            model=uncertainty_adapter,
        )
    elif uncertainty_sampler_config.type == "cross_bayesian":
        from lerobot.uncertainty.uncertainty_samplers.cross_bayesian_sampler import (
            CrossBayesianSampler,
        )

        if uncertainty_sampler_config.cross_bayesian_sampler.scorer_type == "ensemble" and scorer_artifacts.ensemble_model is None:
            raise ValueError(
                "Cross-Bayesian uncertainty sampler with scorer_type=ensemble requires an ensemble model."
            )
        if uncertainty_sampler_config.cross_bayesian_sampler.scorer_type == "laplace" and scorer_artifacts.laplace_posterior is None:
            raise ValueError(
                "Bayesian uncertainty sampler with scorer_type=laplace requires Laplace posterior to draw a scorer model from."
            )

        return CrossBayesianSampler(
            config=uncertainty_sampler_config.cross_bayesian_sampler,
            sampler_model=uncertainty_adapter,
            scorer_artifacts=scorer_artifacts,
        )
    elif uncertainty_sampler_config.type == "entropy":
        from lerobot.uncertainty.uncertainty_samplers.entropy_sampler import (
            EntropySampler,
        )

        return EntropySampler(
            config=uncertainty_sampler_config.entropy_sampler,
            model=uncertainty_adapter,
        )
    else:
        raise ValueError(f"Unknown uncertainty sampler {uncertainty_sampler_config.type}.")

def make_uncertainty_scoring_metric(
    config: ScoringMetricConfig,
    uncertainty_sampler: UncertaintySampler | None = None,
) -> UncertaintyMetric:
    if config.metric_type == "inter_vel_diff":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import InterVelDiff

        return InterVelDiff(
            config=config,
            uncertainty_sampler=uncertainty_sampler
        )
    elif config.metric_type == "likelihood":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import Likelihood

        return Likelihood(
            config=config,
            uncertainty_sampler=uncertainty_sampler,
        )
    elif config.metric_type == "mode_distance":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import ModeDistance

        return ModeDistance(config=config)
    elif config.metric_type == "terminal_vel_norm":
        from lerobot.uncertainty.uncertainty_scoring.scoring_metrics import TerminalVelNorm

        return TerminalVelNorm(config=config)
    else:
        raise ValueError(f"Unknown scoring metric type: {config.metric_type}.")
