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
from typing import Optional, Sequence

from lerobot.common import (
    policies,  # noqa: F401
)
from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.common.datasets.video_utils import get_safe_default_codec
from lerobot.common.envs.configs import PerturbationConfig


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
    # Which uncertainty estimation methods to evaluate
    uncert_est_methods: list[str] = field(
        default_factory=lambda: [
            "composed_likelihood",
            "cross_likelihood",
            "likelihood",
            "epsilon_ball",
        ]
    )

    perturb_types: list[str] = field(
        default_factory=lambda: ["static", "dynamic"]
    )

    @property
    def perturbation_configs(self) -> dict[str, PerturbationConfig]:
        default_perturb_cfg = PerturbationConfig()
        perturbation_configs = {}
        for t in self.perturb_types:
            perturbation_configs[t] = PerturbationConfig(
                enable = True,
                static = (t == "static"),
                min_frac = default_perturb_cfg.min_frac,
                max_frac = default_perturb_cfg.max_frac,
            )
        return perturbation_configs
    
    def validate(self):
        allowed_methods = {
            "composed_likelihood",
            "cross_likelihood",
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

        allowed_perturbs = {"clean", "static", "dynamic"}
        for p in self.perturb_types:
            if p not in allowed_perturbs:
                raise ValueError(
                    f"Unknown perturbation type '{p}'. "
                    f"Allowed: {sorted(allowed_perturbs)}"
                )

@dataclass
class VisConfig:
    """Options that control what we draw and how we save/show it."""
    # Which visualizers to run (you can pick one or more)
    vis_types: list[str] = field(
        default_factory=lambda: ["action_seq", "flows"]
    )

    # If set, start drawing only after this environment step
    start_step: Optional[int] = None

    # Hard cap on how many env steps to visualise
    max_steps: Optional[int] = None

    # Names of the action dimensions to visualize
    action_dim_names: Optional[Sequence[str]] = field(default_factory=lambda: ["x", "y"])

    # Parameters for action sequence visualization
    num_action_seq: int = 30

    # Parameters for flows visualization
    action_dims: Sequence[int] = (0,1)
    action_steps: Optional[list[int]] = None
    num_paths: int = 50

    # Parameters for vector field visualization
    min_action: Sequence[float] = (-1.0, -1.0)
    max_action: Sequence[float] = (1.0, 1.0)
    grid_size: int = 50
    time_grid: Optional[list[float]] = None