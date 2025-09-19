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
import importlib

import gymnasium as gym

from lerobot.envs.configs import AlohaEnv, EnvConfig, PushtEnv, XarmEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")

def make_single_env(
    cfg: EnvConfig, seed: int | None = None
) -> gym.Env:
    package_name = f"gym_{cfg.type}"
    
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e
    
    gym_handle = f"{package_name}/{cfg.task}"
    
    # Set the random number generator of the Libero env to deterministically choose a task ID
    if cfg.task == "LiberoEnv-v0":
        cfg.set_task_sampling_seed(seed)

    gym_kwargs = cfg.gym_kwargs

    # Edit the environment kwargs before creation if OoD is enabled
    gym_kwargs = cfg.ood.tweak_gym_kwargs(gym_kwargs)

    env = gym.make(gym_handle, disable_env_checker=True, **gym_kwargs)

    # Wrap the environment if OoD is enabled
    env = cfg.ood.wrap(env)

    return env

def make_env(
    cfg: EnvConfig,
    n_envs: int = 1,
    use_async_envs: bool = False,
    seeds: list[int] | None = None
) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the config.

    Args:
        cfg (EnvConfig): the config of the environment to instantiate.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed

    Returns:
        gym.vector.VectorEnv: The parallelized gym.env instance.
    """
    if n_envs < 1:
        raise ValueError("`n_envs must be at least 1")
    
    if seeds is not None and len(seeds) != n_envs:
        raise ValueError(
            f"Length of seed list must equal n_envs ({n_envs}), but got length {len(seeds)}."
        )

    package_name = f"gym_{cfg.type}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e
    
    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    if seeds is None:    
        env_fns = [
            lambda cfg=cfg: make_single_env(cfg) for _ in range(n_envs)
        ]
    else:
        env_fns = [
            lambda cfg=cfg, seed=seeds[i]: make_single_env(cfg, seed=seed)
                for i in range(n_envs)
        ]
    env = env_cls(env_fns)

    return env