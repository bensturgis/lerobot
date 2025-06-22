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

import logging

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Sequence, Union

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import env_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.configuration_uncertainty_sampler import UncertaintySamplerConfig
from lerobot.common.policies.flow_matching.estimate_uncertainty import FlowMatchingUncertaintySampler
from lerobot.common.policies.flow_matching.visualizer import (
    FlowMatchingVisualizer,
    ActionSeqVisualizer,
    FlowVisualizer,
    VectorFieldVisualizer,
)
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.default import VisConfig
from lerobot.configs.types import FeatureType


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "flow_matching":
        from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy

        return FlowMatchingPolicy
    elif name == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
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
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_rgb_encoder(cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata):
    features = dataset_to_policy_features(ds_meta.features)
    cfg.input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
    
    policy_type = cfg.type
    if policy_type == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
        
        return DiffusionRgbEncoder(cfg)
    elif policy_type == "flow_matching":
        from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingRgbEncoder

        return FlowMatchingRgbEncoder(cfg)
    elif policy_type == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTRgbEncoder

        return VQBeTRgbEncoder(cfg)
    else:
        raise ValueError(f"Policy type '{policy_type}' does not have a RGB encoder.")


def make_flow_matching_uncertainty_sampler(
    flow_matching_cfg: FlowMatchingConfig,
    uncertainty_sampler_cfg: UncertaintySamplerConfig,
    flow_matching_model: nn.Module,
    scorer_flow_matching_model: Optional[nn.Module] = None,
    laplace_calib_loader: Optional[DataLoader] = None,
    laplace_path: Optional[Path] = None,
) -> FlowMatchingUncertaintySampler:
    if uncertainty_sampler_cfg.type == "composed_cross_ensemble":
        from lerobot.common.policies.flow_matching.estimate_uncertainty import ComposedCrossEnsembleSampler

        return ComposedCrossEnsembleSampler(
            flow_matching_cfg=flow_matching_cfg,
            cfg=uncertainty_sampler_cfg.composed_cross_ensemble_sampler,
            sampler_flow_matching_model=flow_matching_model,
            scorer_flow_matching_model=scorer_flow_matching_model
        )
    if uncertainty_sampler_cfg.type == "composed_cross_laplace":
        from lerobot.common.policies.flow_matching.estimate_uncertainty import ComposedCrossLaplaceSampler

        return ComposedCrossLaplaceSampler(
            flow_matching_cfg=flow_matching_cfg,
            cfg=uncertainty_sampler_cfg.composed_cross_laplace_sampler,
            flow_matching_model=flow_matching_model,
            laplace_calib_loader=laplace_calib_loader,
            laplace_path=laplace_path,
        )
    if uncertainty_sampler_cfg.type == "composed_sequence":
        from lerobot.common.policies.flow_matching.estimate_uncertainty import ComposedSequenceSampler

        return ComposedSequenceSampler(
            flow_matching_cfg=flow_matching_cfg, 
            cfg=uncertainty_sampler_cfg.composed_sequence_sampler,
            velocity_model=flow_matching_model.unet
        )
    elif uncertainty_sampler_cfg.type == "cross_ensemble":
        from lerobot.common.policies.flow_matching.estimate_uncertainty import CrossEnsembleSampler

        if scorer_flow_matching_model is None:
            raise ValueError("Ensemble cross-likelihood uncertainty sampler requires a scorer model.")
        return CrossEnsembleSampler(
            flow_matching_cfg=flow_matching_cfg,
            cfg=uncertainty_sampler_cfg.cross_ensemble_sampler,
            sampler_flow_matching_model=flow_matching_model,
            scorer_flow_matching_model=scorer_flow_matching_model,
        )
    elif uncertainty_sampler_cfg.type == "cross_laplace":
        from lerobot.common.policies.flow_matching.estimate_uncertainty import CrossLaplaceSampler

        if laplace_path is None:
            raise ValueError(
                "Laplace cross-likelihood uncertainty sampler requires a path to save/load the "
                "Laplace posterior"
                )
        if laplace_calib_loader is None and not laplace_path.exists():
            raise ValueError(
                "Laplace cross-likelihood uncertainty sampler requires a calibration data "
                "to fit the Laplace posterior."
            )
        return CrossLaplaceSampler(
            flow_matching_cfg=flow_matching_cfg,
            cfg=uncertainty_sampler_cfg.cross_laplace_sampler,
            flow_matching_model=flow_matching_model,
            laplace_calib_loader=laplace_calib_loader,
            laplace_path=laplace_path,
        )
    elif uncertainty_sampler_cfg.type == "likelihood":
        from lerobot.common.policies.flow_matching.estimate_uncertainty import LikelihoodSampler

        return LikelihoodSampler(
            flow_matching_cfg=flow_matching_cfg,
            cfg=uncertainty_sampler_cfg.likelihood_sampler,
            velocity_model=flow_matching_model.unet
        )
    elif uncertainty_sampler_cfg.type == "epsilon_ball":
        from lerobot.common.policies.flow_matching.estimate_uncertainty import EpsilonBallSampler

        return EpsilonBallSampler(
            flow_matching_cfg=flow_matching_cfg,
            cfg=uncertainty_sampler_cfg.epsilon_ball_sampler,
            velocity_model=flow_matching_model.unet
        )
    else:
        raise ValueError(f"Unknown uncertainty sampler {uncertainty_sampler_cfg.type}.")


def make_flow_matching_visualizers(
    vis_cfg: VisConfig,
    model_cfg: FlowMatchingConfig,
    velocity_model: nn.Module,
    output_root: Optional[Union[str, Path]] = None,
    unnormalize_outputs: Optional[nn.Module] = None,
) -> Sequence[FlowMatchingVisualizer]:
    """
    Factory that picks and constructs the right flow matching visualizer
    subclasses based on vis_cfg.vis_types.
    """
    common_kwargs = dict(
        config = model_cfg,
        velocity_model = velocity_model,
        action_dim_names = vis_cfg.action_dim_names,
        show = vis_cfg.show,
        save = True,
        output_root = output_root,
        create_gif = True,
        verbose = False,
    )

    visualizers = []
    for vis_type in vis_cfg.vis_types:
        if vis_type == "flows":
            visualizers.append(
                FlowVisualizer(
                    **common_kwargs,
                    action_dims = vis_cfg.action_dims,
                    axis_limits=vis_cfg.axis_limits,
                    action_steps = vis_cfg.action_steps,
                    num_paths = vis_cfg.num_paths,
                )
            )
        elif vis_type == "vector_field":
            visualizers.append(
                VectorFieldVisualizer(
                    **common_kwargs,
                    action_dims = vis_cfg.action_dims,
                    action_steps = vis_cfg.action_steps,
                    min_action = vis_cfg.min_action,
                    max_action = vis_cfg.max_action,
                    grid_size = vis_cfg.grid_size,
                    time_grid = vis_cfg.time_grid,
                )
            )
        elif vis_type == "action_seq":
            if unnormalize_outputs is None:
                raise ValueError(
                    "Require `unnormalize_outputs` module to create ActionSeqVisualizer "
                    "to map normalized actions back to the original scale."
                )
            
            visualizers.append(
                ActionSeqVisualizer(
                    **common_kwargs,
                    unnormalize_outputs=unnormalize_outputs,
                    num_action_seq=vis_cfg.num_action_seq,
                )
            ) 
        else:
            raise ValueError(f"Unknown visualization type {vis_type}.")
        
    return visualizers


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    uncertainty_sampler_cfg: UncertaintySamplerConfig | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
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
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.type == "flow_matching":
        kwargs["uncertainty_sampler_config"] = uncertainty_sampler_cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
