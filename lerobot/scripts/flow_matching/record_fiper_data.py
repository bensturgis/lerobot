import logging
import random
import time
from itertools import cycle
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy
from lerobot.common.policies.flow_matching.uncertainty.utils.scorer_artifacts import (
    build_scorer_artifacts_for_fiper_recorder,
)
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import save_episode_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.configs.fiper_data_recording import FiperDataRecordingPipelineConfig


def rollout(
    env: gym.Env,
    policy: FlowMatchingPolicy,
    seed: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    device = get_device_from_parameters(policy)

    # Initialize random number generator to deterministically select actions
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    if hasattr(env, "camera_names") and env.camera_names is not None:
        ep_frames: Dict[str, list[np.ndarray]] = {
            cam: [] for cam in env.camera_names
        }
    else:
        ep_frames: list[np.ndarray] = []

    success = False

    start_time = time.time()

    # Reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=seed)

    if hasattr(env, "camera_names") and env.camera_names is not None:
        for camera in env.camera_names:
            ep_frames[camera].append(env.unwrapped.render(camera_name=camera))
    else:
        ep_frames.append(env.render())

    max_episode_steps = env.spec.max_episode_steps
    progbar = trange(
        max_episode_steps,
        desc=f"Running rollout with at most {max_episode_steps} steps."
    )
    for _ in progbar:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        with torch.no_grad():
            action = policy.select_action(observation, generator)

        # Apply the next action
        observation, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
        if hasattr(env, "camera_names") and env.camera_names is not None:
            for camera in env.camera_names:
                ep_frames[camera].append(env.unwrapped.render(camera_name=camera))
        else:
            ep_frames.append(env.render())

        if info is not None and "is_success" in info:
            success = bool(info["is_success"])

        # Stop early if environment terminates
        done = terminated or truncated
        if done:
            break

    logging.info(f"Finished episode in {time.time() - start_time:.1f}s")

    info = {
        "successful": success,
    }

    return info, ep_frames

def choose_seed(rng: Optional[random.Random] = None) -> int:
    """
    Return a fresh 32-bit random seed using a random number generator.
    """
    return rng.randrange(2**31 - 1)

def prepare_run_metadata(cfg: FiperDataRecordingPipelineConfig) -> Dict[str, Any]:
    """
    Prepare metadata dictionary for this data recording run.
    """
    run_metadata = {
        "metadata": True,
        "task": cfg.env.task,
        "action_prediction_horizon": cfg.policy.horizon,
        "action_execution_horizon": cfg.policy.n_action_steps,
        "action_batch_size": cfg.fiper_data_recorder.num_uncertainty_sequences,
    }

    return run_metadata

@parser.wrap()
def main(cfg: FiperDataRecordingPipelineConfig): 
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Random number generator to choose seeds for the single runs
    rng = random.Random(cfg.seed)

    if cfg.policy.type != "flow_matching":
        raise ValueError(
            f"FIPER data recording only supported for Flow Matching policies, "
            f"but got policy type '{cfg.policy.type}'."
        )

    run_metadata = prepare_run_metadata(cfg)

    logging.info("Loading policy")
    device = get_safe_torch_device(cfg.policy.device, log=True)
    policy: FlowMatchingPolicy = make_policy(
        cfg.policy,
        env_cfg=cfg.env,
    ).to(device)
    policy.eval()
    
    scorer_artifacts = build_scorer_artifacts_for_fiper_recorder(
        fiper_data_recorder_cfg=cfg.fiper_data_recorder,
        policy_cfg=cfg.policy,
        env_cfg=cfg.env,
        dataset_cfg=cfg.dataset,
        policy=policy,
    )
    policy.init_fiper_data_recorder(
        config=cfg.fiper_data_recorder,
        scorer_artifacts=scorer_artifacts,
    )

    is_ood_cycle = cycle([False, True])

    n_episodes = cfg.n_episodes
    progbar = trange(
        n_episodes,
        desc=f"Recording FIPER data for {n_episodes} episodes."
    )
    for episode in progbar:
        is_ood = next(is_ood_cycle)
        distribution_label = "OOD" if is_ood else "ID"
        logging.info(f"Creating {distribution_label} environment.")
        seed = choose_seed(rng)
        cfg.env.ood.enabled = is_ood
        env = make_single_env(cfg.env, seed)
        rollout_info, ep_frames = rollout(
            env=env,
            policy=policy,
            seed=seed,
        )

        if cfg.save_videos:
            save_episode_video(
                ep_frames=ep_frames,
                out_root=cfg.output_dir / "videos",
                episode_idx=episode + 1,
                fps=env.metadata["render_fps"],
            )

        ep_metadata = {
            **run_metadata,
            **rollout_info,
            "episode": episode,
        }

        policy.fiper_data_recorder.save_data(
            output_dir=cfg.output_dir,
            episode_metadata=ep_metadata,
        )

        policy.fiper_data_recorder.reset()
        

if __name__ == "__main__":
    init_logging()
    main()