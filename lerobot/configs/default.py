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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot.common import (
    policies,  # noqa: F401
)
from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.common.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)


@dataclass
class WandBConfig:
    enable: bool = True
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = True
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False
    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )

@dataclass
class EvalUncertEstConfig:
    n_episodes: int = 20
    
    # If True we ignore success/failure and only keep ID vs. OoD
    collapse_success_failure: bool = False

    # Which uncertainty estimation methods to evaluate
    uncert_est_methods: list[str] = field(
        default_factory=lambda: [
            "composed_sequence",
            "composed_cross_ensemble",
            "composed_cross_laplace"
            "cross_ensemble",
            "cross_laplace"
            "likelihood",
            "epsilon_ball",
        ]
    )

    # Paths to ID and OoD failure seeds to balance success and failure cases
    id_failure_seeds_path: Path | None = None
    ood_failure_seeds_path: Path | None = None
    
    def validate(self):
        allowed_methods = {
            "composed_sequence",
            "composed_cross_ensemble",
            "composed_cross_laplace"
            "cross_ensemble",
            "cross_laplace",
            "likelihood",
            "epsilon_ball",
        }
        # check every method the user passed
        for m in self.uncert_est_methods:
            if m not in allowed_methods:
                raise ValueError(
                    f"Unknown uncertainty-estimation method '{m}'. "
                    f"Allowed: {sorted(allowed_methods)}"
                )

@dataclass
class FlowVisConfig:
    # Two or three indices indicating which action dimensions to visualize
    action_dims: list[int] = (0,1)

    # Names of the action dimensions to visualize
    action_dim_names: Optional[list[str]] = None

    # Time‐step indices (horizon steps) at which to generate visualizations
    action_steps: Optional[list[int]] = None

    # Whether to display plots live
    show: bool = False

    # Custom axis limits for each plotted dimension as a list of (min, max) tuples
    axis_limits: Optional[list[tuple[float, float]]] = None

    # Number of trajectory samples to draw when visualizing multiple action sequences
    num_paths: int = 50

@dataclass
class VectorFieldVisConfig:
    # Two or three indices indicating which action dimensions to visualize
    action_dims: list[int] = (0,1)

    # Names of the action dimensions to visualize
    action_dim_names: Optional[list[str]] = None

    # Time‐step indices (horizon steps) at which to generate visualizations
    action_steps: Optional[list[int]] = None

    # Whether to display plots live
    show: bool = False

    # Minimum value of the action space (default −1.0 after normalization)
    min_action: float = -1.0

    # Maximum value of the action space (default +1.0 after normalization)
    max_action: float = 1.0

    # Number of grid points per axis when sampling the action space for the quiver plot
    grid_size: int = 50

    # List of time values (between 0 and 1) at which to compute and draw the vector field
    time_grid: Optional[list[float]] = None

@dataclass
class ActionSeqVisConfig:
    # Whether to display plots live
    show: bool = False

    # Parameters for action sequence visualization
    num_action_seq: int = 30

@dataclass
class VisConfig:
    """Options that control what we draw and how we save/show it."""
    # Number of total rollouts to visualize
    num_rollouts: int = 10

    # If set, start drawing only after this environment step
    start_step: Optional[int] = None

    # Hard cap on how many env steps to visualize
    max_steps: Optional[int] = None

    # Names of the action dimensions to visualize
    action_dim_names: Optional[list[str]] = None
