#!/usr/bin/env python
"""
Evaluate a Flow Matching policy's uncertainty estimates under clean and perturbed conditions.

Runs rollouts with multiple uncertainty estimation methods, records per-step uncertainty scores,
saves rollout videos, and generates comparison plots for each method and perturbation.

Usage example:

You want to plot uncertainty estimation scores for the Action Sequence Likelihood and the Composed 
Action Sequence Likelihood methods as well as for static and dynamic perturbations for 10 episodes:

```
python lerobot/scripts/eval_uncertainty_estimation.py \
    --policy.path=outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht \
    --eval_uncert_est.n_episodes=10 \
    --eval_uncert_est.uncert_est_methods='["action_seq_likelihood", "composed_action_seq_likelihood"]'
```

"""
import logging
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval_uncertainty_estimation import EvalUncertaintyEstimationPipelineConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

def plot_episode_uncertaintes(
    episode: int,
    uncert_est_method: str,
    perturb_type: str,
    clean_uncertainties: np.ndarray,
    perturb_uncertainties: np.ndarray,
    output_dir: Path,
):
    clean_uncertainties = np.where(np.isneginf(clean_uncertainties), np.nan, clean_uncertainties)
    perturb_uncertainties = np.where(np.isneginf(perturb_uncertainties), np.nan, perturb_uncertainties)

    plt.figure()
    plt.plot(
        np.arange(len(clean_uncertainties)), 
        clean_uncertainties, 
        label="clean"
    )
    plt.plot(
        np.arange(len(perturb_uncertainties)), 
        perturb_uncertainties, 
        label=perturb_type
    )
    plt.xlabel("Action-Sequence Index")
    plt.ylabel("Uncertainty Score")
    plt.title(f"{uncert_est_method.replace('_', ' ').title()} - Episode {episode + 1}")
    plt.legend()
    file_path = output_dir / f"uncertainty_scores_ep{(episode + 1):03d}.png"
    plt.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_all_uncertainties(
    uncert_est_method: str,
    perturb_type: str,
    clean_uncertainties: list[np.ndarray],
    perturb_uncertainties: list[np.ndarray],
    output_dir: Path,
):
    # Pick consistent colors
    clean_uncert_color = "C0"
    perturb_uncert_color  = "C1"

    plt.figure()

    # Plot all clean episodes in blue
    for episode, uncertainties in enumerate(clean_uncertainties):
        uncertainties = np.where(np.isneginf(uncertainties), np.nan, uncertainties)
        action_seq_indices = np.arange(len(uncertainties))
        plt.plot(
            action_seq_indices,
            uncertainties,
            color=clean_uncert_color,
            alpha=0.5, 
            label="clean" if episode == 0 else None
        )

    # Plot all perturbed episodes in orange
    for episode, uncertainties in enumerate(perturb_uncertainties):
        uncertainties = np.where(np.isneginf(uncertainties), np.nan, uncertainties)
        action_seq_indices = np.arange(len(uncertainties))
        plt.plot(
            action_seq_indices,
            uncertainties, 
            color=perturb_uncert_color,
            alpha=0.5, 
            label=perturb_type if episode == 0 else None
        )

    # Determine maximum episode length
    max_episode_len = max(len(u) for u in clean_uncertainties + perturb_uncertainties)

    def compute_uncert_stats(uncertainties):
        uncerts_mean, uncerts_std = [], []
        for action_seq_idx in np.arange(max_episode_len):
            uncerts_at_idx = np.array([
                u[action_seq_idx]
                for u in uncertainties
                if len(u) > action_seq_idx
                and np.isfinite(u[action_seq_idx])
            ])
            if uncerts_at_idx.size > 0:
                uncerts_mean.append(uncerts_at_idx.mean())
                uncerts_std.append(uncerts_at_idx.std())
            else:
                uncerts_mean.append(np.nan)
                uncerts_std.append(np.nan)
        return np.array(uncerts_mean), np.array(uncerts_std)

    clean_uncerts_mean, clean_uncerts_std = compute_uncert_stats(clean_uncertainties)
    perturb_uncerts_mean, perturb_uncerts_std = compute_uncert_stats(perturb_uncertainties)

    # Plot means and standard deviation bands
    plt.plot(
        np.arange(max_episode_len),
        clean_uncerts_mean,
        color=clean_uncert_color,
    )
    plt.fill_between(
        np.arange(max_episode_len),
        clean_uncerts_mean - clean_uncerts_std,
        clean_uncerts_mean + clean_uncerts_std,
        alpha=0.2
    )
    plt.plot(
        np.arange(max_episode_len),
        perturb_uncerts_mean,
        color=perturb_uncert_color,
    )
    plt.fill_between(
        np.arange(max_episode_len),
        perturb_uncerts_mean - perturb_uncerts_std,
        perturb_uncerts_mean + perturb_uncerts_std,
        alpha=0.2
    )

    plt.xlabel("Action-Sequence Index")
    plt.ylabel("Uncertainty Score")
    plt.title(f"{uncert_est_method.replace('_', ' ').title()}")
    plt.legend()

    file_path = output_dir / f"uncertainty_scores_{uncert_est_method}.png"
    plt.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close()


def rollout(
    env: gym.Env,
    policy: PreTrainedPolicy,
) -> Dict[str, np.ndarray]:
    device = get_device_from_parameters(policy)

    ep_uncertainties = []
    ep_frames = []

    start_time = time.time() 

    # Reset the policy and environments.
    policy.reset()
    observation, _ = env.reset()

    ep_frames.append(env.render())
    
    max_episode_steps = env.spec.max_episode_steps
    # max_episode_steps = 30
    progbar = trange(
        max_episode_steps,
        desc=f"Running rollout with at most {max_episode_steps} steps."
    ) 
    for step_idx in progbar:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        # Decide whether a new sequence will be generated
        new_action_gen = len(policy._queues["action"]) == 0

        with torch.no_grad():
            action = policy.select_action(observation)

        if new_action_gen:
            uncertainty = policy.flow_matching.uncertainty_sampler.latest_uncertainties.detach().cpu().mean().item()
            
            ep_uncertainties.append(uncertainty)

        # Apply the next action
        observation, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        ep_frames.append(env.render())

        # Stop early if environment terminates
        done = terminated or truncated
        if done:
            break

    logging.info(f"Finished episode in {time.time() - start_time:.1f}s")

    info = {
        "ep_uncertainties": ep_uncertainties,
        "ep_frames": ep_frames,
    }

    return info

@parser.wrap()
def main(cfg: EvalUncertaintyEstimationPipelineConfig): 
    if cfg.policy.type != "flow_matching":
        raise ValueError(
            f"visualize_flow_matching.py only supports Flow Matching policies, "
            f"but got policy type '{cfg.policy.type}'."
        )
    
    n_episods = cfg.eval_uncert_est.n_episodes
    all_uncertanties = defaultdict(lambda: defaultdict(list))
    progbar = trange(
        n_episods,
        desc=f"Evaluating uncertainty estimation for {n_episods} episodes."
    )   
    for episode in progbar:
        for uncert_est_method in cfg.eval_uncert_est.uncert_est_methods:
            logging.info("Loading policy")
            cfg.policy.sample_with_uncertainty = True
            cfg.policy.uncertainty_sampler = uncert_est_method
            device = get_safe_torch_device(cfg.policy.device, log=True)
            policy = make_policy(cfg.policy, env_cfg=cfg.env).to(device)
            policy.eval()
            
            logging.info(f"Creating clean environment.")
            cfg.env.perturbation.enable = False
            clean_env = make_single_env(cfg.env)
            clean_ep_info = rollout(clean_env, policy)
            
            all_uncertanties[uncert_est_method]["clean"].append(clean_ep_info["ep_uncertainties"])
            clean_output_dir = cfg.output_dir / uncert_est_method / "clean"
            clean_output_dir.mkdir(parents=True, exist_ok=True)
            write_video(
                str(clean_output_dir / f"rollout_ep{(episode + 1):03d}.mp4"),
                np.stack(clean_ep_info["ep_frames"], axis=0),
                fps=clean_env.metadata["render_fps"]
            )
            for perturb_type, perturb_cfg in cfg.eval_uncert_est.perturbation_configs.items():
                logging.info(f"Creating environment with {perturb_type} perturbation.")
                perturb_env = make_single_env(replace(cfg.env, perturbation=perturb_cfg))
                perturb_ep_info = rollout(perturb_env, policy)
                
                all_uncertanties[uncert_est_method][perturb_type].append(perturb_ep_info["ep_uncertainties"])
                perturb_output_dir = cfg.output_dir / uncert_est_method / perturb_type
                perturb_output_dir.mkdir(parents=True, exist_ok=True)
                write_video(
                    str(perturb_output_dir / f"rollout_ep{(episode + 1):03d}.mp4"),
                    np.stack(perturb_ep_info["ep_frames"], axis=0),
                    fps=perturb_env.metadata["render_fps"]
                )
                plot_episode_uncertaintes(
                    episode=episode,
                    uncert_est_method=uncert_est_method,
                    perturb_type=perturb_type,
                    clean_uncertainties=clean_ep_info["ep_uncertainties"],
                    perturb_uncertainties=perturb_ep_info["ep_uncertainties"],
                    output_dir=perturb_output_dir,
                )
                plot_all_uncertainties(
                    uncert_est_method=uncert_est_method,
                    perturb_type=perturb_type,
                    clean_uncertainties=all_uncertanties[uncert_est_method]["clean"],
                    perturb_uncertainties=all_uncertanties[uncert_est_method][perturb_type],
                    output_dir=perturb_output_dir,
                )

if __name__ == "__main__":
    init_logging()
    main()