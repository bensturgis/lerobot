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
import random
from pprint import pformat
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch.utils.data import Subset

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.eval_uncertainty_estimation import EvalUncertaintyEstimationPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(
    cfg: Union[TrainPipelineConfig, EvalUncertaintyEstimationPipelineConfig]
) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


class TrainValSplit(NamedTuple):
    train_dataset: Subset
    val_dataset: Subset
    train_ep_ids: set[int]
    val_ep_ids: set[int]


def _split_indices_by_episode(
    frame_idxs_per_ep: dict[int, list[int]],
    val_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[list[int], list[int]]:
    """
    Split frame indices so that no episode is shared between train and val.

    Args:
        frame_idxs_per_ep: Mapping from each episode ID to the list of frame
            indices belonging to it.
        val_ratio: Fraction of episodes to reserved for validation.
        seed: Random seed for shuffling episodes.

    Returns:
        Frame indices for the training and validation subset.
    """
    # Collect and shuffle episode IDs
    episode_ids = list(frame_idxs_per_ep.keys())
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(episode_ids)
    else:
        random.shuffle(episode_ids)

    num_val_eps = max(1, int(len(episode_ids) * val_ratio))
    val_ep_ids = set(episode_ids[:num_val_eps])
    train_ep_ids = set(episode_ids[num_val_eps:])

    train_indices = [
        idx
        for ep_id in train_ep_ids
        for idx in frame_idxs_per_ep[ep_id]
    ]
    val_indices = [
        idx
        for ep_id in val_ep_ids
        for idx in frame_idxs_per_ep[ep_id]
    ]
    return train_indices, val_indices, train_ep_ids, val_ep_ids


def make_train_val_split(
    full_dataset: LeRobotDataset, cfg: TrainPipelineConfig
) -> Tuple[Subset, Subset]:
    """
    Load a full dataset and return disjoint train/validation subsets by episode.

    Args:
        full_dataset: The complete dataset containing all episodes and frames.
        cfg: A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Returns:
        Training and validation dataset containing separate episodes.
    """    
    # Build a mapping from episode ID to a list of frame indices
    frame_idxs_per_ep: Dict[int, List[int]] = {}
    ep_starts = full_dataset.episode_data_index["from"].tolist()
    ep_ends = full_dataset.episode_data_index["to"].tolist()
    for ep_id, (ep_start_idx, ep_end_idx) in enumerate(zip(ep_starts, ep_ends)):
        frame_idxs_per_ep[ep_id] = list(range(ep_start_idx, ep_end_idx))

    # Split into two flat index lists, using cfg.val_ratio
    train_idxs, val_idxs, train_ep_ids, val_ep_ids = _split_indices_by_episode(
        frame_idxs_per_ep,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed or None,
    )

    return TrainValSplit(
        train_dataset=Subset(full_dataset, train_idxs),
        val_dataset=Subset(full_dataset, val_idxs),
        train_ep_ids=train_ep_ids,
        val_ep_ids=val_ep_ids 
    )