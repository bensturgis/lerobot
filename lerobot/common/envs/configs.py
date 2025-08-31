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

import abc
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import draccus
import gymnasium as gym
from libero.libero import benchmark as lb_bench

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.common.envs.wrappers import PerturbationWrapper
from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class OODConfig(abc.ABC):
    enabled: bool = False

    def tweak_gym_kwargs(self, kwargs: dict) -> dict:
        """Adjust kwargs for gym.make()."""
        return self._tweak_gym_kwargs_impl(kwargs) if self.enabled else kwargs

    def _tweak_gym_kwargs_impl(self, kwargs: dict) -> dict:
        """Will be overwritten by subclasses."""
        return kwargs
    
    def wrap(self, env: gym.Env) -> gym.Env:
        """Return wrapped env."""
        return self._wrap_impl(env) if self.enabled else env

    def _wrap_impl(self, env: gym.Env) -> gym.Env:
        """Will be overwritten by subclasses."""
        return env


@dataclass
class ImagePatchOODConfig(OODConfig):
    static: bool = True
    min_frac: float = 0.1
    max_frac: float = 0.2
    allowed_area: Tuple[float, float] | None = None
    patch_color: Tuple[int, int, int] = (0, 0, 0)

    def _tweak_gym_kwargs_impl(self, kwargs: dict) -> dict:
        return kwargs

    def _wrap_impl(self, env: gym.Env) -> gym.Env:        
        return PerturbationWrapper(
            env,
            static=self.static,
            min_patch_frac=self.min_frac,
            max_patch_frac=self.max_frac,
            allowed_area=self.allowed_area,
            patch_color=self.patch_color,
        )
    

@dataclass
class BddlSwapOODConfig(OODConfig):
    bddl_root: Path = Path(__file__).resolve().parent  / "libero_bddl_files"
    
    def _tweak_gym_kwargs_impl(self, kwargs: dict) -> dict:
        # Exchange in-distribution BDDL file by its corresponding out-of-distribution BDDL file
        id_bddl_path = kwargs["bddl_file_name"]
        bddl_filename = id_bddl_path.name
        ood_bddl_path = id_bddl_path.parent.parent / "ood" / bddl_filename
        kwargs["bddl_file_name"] = ood_bddl_path

        return kwargs

    def _wrap_impl(self, env: gym.Env) -> gym.Env:
        return env


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    ood: OODConfig = OODConfig()

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    # Out-of-distribution configuration
    ood: ImagePatchOODConfig = ImagePatchOODConfig()

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }

@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    benchmark: str = "libero_90"
    bddl_root: Path = Path(__file__).resolve().parent  / "libero_bddl_files"
    task: str = "LiberoEnv-v0"
    task_ids: List[int] = field(default_factory=list)
    task_sample_seed: int | None = None
    fps: int = 20
    episode_length: int = 200
    # Camera and simulation settings
    robots: list[str] = field(default_factory=lambda: ["Panda"])
    camera_heights: int = 256
    camera_widths: int = 256
    control_freq: int = 20
    horizon: int = 1000
    has_renderer: bool = False       # True if you want on-screen
    render_camera: str  = "frontview"
    camera_names: list[str] = field(default_factory=lambda: [
        "frontview", "agentview", "robot0_eye_in_hand"
    ])
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "pixels/image": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
            "pixels/wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3))
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels/image": f"{OBS_IMAGES}.image",
            "pixels/wrist_image": f"{OBS_IMAGES}.wrist_image",
        }
    )
    # Out-of-distribution configuration
    ood: BddlSwapOODConfig = BddlSwapOODConfig()

    def __post_init__(self):
        self.task_id_rng = random.Random(self.task_sample_seed)

    def set_task_sampling_seed(self, seed: int) -> None:
        """Seed the RNG that picks a Libero task ID."""
        self.task_sample_seed = seed
        self.task_id_rng = random.Random(seed)

    @property
    def gym_kwargs(self) -> dict:
        benchmark_dict = lb_bench.get_benchmark_dict()[self.benchmark]()
        
        # Choose an a task ID
        all_task_ids = list(range(benchmark_dict.get_num_tasks()))
        if not self.task_ids:
            self.task_ids = all_task_ids
        chosen_task_id = self.task_id_rng.choice(self.task_ids)

        if chosen_task_id not in all_task_ids:
            raise ValueError(
                f"Task ID {chosen_task_id} is invalid for benchmark '{self.benchmark}' "
                f"(valid range 0 … {benchmark_dict.get_num_tasks()-1})."
            )
        
        # Get the path to the corresponding bddl file
        task = benchmark_dict.get_task(chosen_task_id)
        bddl_path = self.bddl_root / self.benchmark / "id" / f"{task.name}.bddl"

        return {
            "bddl_file_name": bddl_path,
            "robots": self.robots,
            "render_camera": self.render_camera,
            "has_renderer": self.has_renderer,
            "camera_names": self.camera_names,
            "control_freq": self.control_freq,
            "camera_heights": self.camera_heights,
            "camera_widths": self.camera_widths,
            "horizon": self.horizon,
            "max_episode_steps": self.episode_length,
        }

@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "environment_state": OBS_ENV,
            "pixels": OBS_IMAGE,
        }
    )

    # Out-of-distribution configuration
    ood: ImagePatchOODConfig = ImagePatchOODConfig()

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }
