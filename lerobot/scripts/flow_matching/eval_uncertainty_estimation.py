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
    --eval_uncert_est.uncert_est_methods='["likelihood", "composed_likelihood"]'
```
"""
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval_uncertainty_estimation import EvalUncertaintyEstimationPipelineConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.flow_matching.laplace_utils import (
    create_laplace_flow_matching_calib_loader,
    make_laplace_path
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

def plot_single_episode_uncertaintes(
    episode: int,
    uncert_est_method: str,
    success_uncertainties: np.ndarray,
    id_success: bool,
    failure_uncertainties: np.ndarray,
    id_failure: bool,
    output_dir: Path,
):
    success_uncertainties = np.where(np.isneginf(success_uncertainties), np.nan, success_uncertainties)
    failure_uncertainties = np.where(np.isneginf(failure_uncertainties), np.nan, failure_uncertainties)

    plt.figure()
    success_label = "ID success" if id_success else "OoD success"
    plt.plot(
        np.arange(len(success_uncertainties)), 
        success_uncertainties, 
        label=success_label
    )
    failure_label = "ID failure" if id_failure else "OoD failure"
    plt.plot(
        np.arange(len(failure_uncertainties)), 
        failure_uncertainties, 
        label=failure_label
    )
    plt.xlabel("Action-Sequence Index")
    plt.ylabel("Uncertainty Score")
    plt.title(f"{uncert_est_method.replace('_', ' ').title()} - Episode {episode + 1}")
    plt.legend()
    file_path = output_dir / f"uncertainty_scores_ep{(episode + 1):03d}.png"
    plt.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_success_failure_uncertainties(
    uncert_est_method: str,
    success_uncertainties: list[np.ndarray],
    id_success: bool,
    failure_uncertainties: list[np.ndarray],
    id_failure: bool,
    output_dir: Path,
):
    # Pick consistent colors
    success_uncert_color = "C0"
    failure_uncert_color = "C1"

    plt.figure()

    # Plot all clean episodes in blue
    for episode, uncertainties in enumerate(success_uncertainties):
        uncertainties = np.where(np.isneginf(uncertainties), np.nan, uncertainties)
        action_seq_indices = np.arange(len(uncertainties))
        success_label = "ID success" if id_success else "OoD success"
        plt.plot(
            action_seq_indices,
            uncertainties,
            color=success_uncert_color,
            alpha=0.3, 
            label=success_label if episode == 0 else None
        )

    # Plot all perturbed episodes in orange
    for episode, uncertainties in enumerate(failure_uncertainties):
        uncertainties = np.where(np.isneginf(uncertainties), np.nan, uncertainties)
        action_seq_indices = np.arange(len(uncertainties))
        failure_label = "ID failure" if id_failure else "OoD failure"
        plt.plot(
            action_seq_indices,
            uncertainties, 
            color=failure_uncert_color,
            alpha=0.3, 
            label=failure_label if episode == 0 else None
        )

    # Determine maximum episode length
    max_episode_len = max(len(u) for u in success_uncertainties + failure_uncertainties)

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

    success_uncerts_mean, success_uncerts_std = compute_uncert_stats(success_uncertainties)
    failure_uncerts_mean, failure_uncerts_std = compute_uncert_stats(failure_uncertainties)

    # Plot means and standard deviation bands
    plt.plot(
        np.arange(max_episode_len),
        success_uncerts_mean,
        color=success_uncert_color,
    )
    plt.fill_between(
        np.arange(max_episode_len),
        success_uncerts_mean - success_uncerts_std,
        success_uncerts_mean + success_uncerts_std,
        alpha=0.5
    )
    plt.plot(
        np.arange(max_episode_len),
        failure_uncerts_mean,
        color=failure_uncert_color,
    )
    plt.fill_between(
        np.arange(max_episode_len),
        failure_uncerts_mean - failure_uncerts_std,
        failure_uncerts_mean + failure_uncerts_std,
        alpha=0.5
    )

    plt.xlabel("Action-Sequence Index")
    plt.ylabel("Uncertainty Score")
    plt.title(f"{uncert_est_method.replace('_', ' ').title()}")
    plt.legend()

    file_path = output_dir / f"uncertainty_scores_{uncert_est_method}.png"
    plt.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_all_uncertainties(
    uncert_est_method: str,
    id_success_uncertainties: list[np.ndarray],
    id_failure_uncertainties: list[np.ndarray],
    ood_success_uncertainties: list[np.ndarray],
    ood_failure_uncertainties: list[np.ndarray],
    output_dir: Path,
):
    """
    Draw per-episode uncertainties and mean ± std bands for
        - ID   success
        - ID   failure
        - OoD  success
        - OoD  failure
    """
    # Consistent colours
    colours = {
        "ID success":  "C0",
        "ID failure":  "C3",
        "OoD success": "C2",
        "OoD failure": "C1",
    }

    # Plot every episode as a faint line
    plt.figure()
    buckets = [
        ("ID success", id_success_uncertainties),
        ("ID failure", id_failure_uncertainties),
        ("OoD success", ood_success_uncertainties),
        ("OoD failure", ood_failure_uncertainties),
    ]

    for label, episodes in buckets:
        for ep_idx, uncert in enumerate(episodes):
            uncert = np.where(np.isneginf(uncert), np.nan, uncert)
            plt.plot(
                np.arange(len(uncert)),
                uncert,
                color=colours[label],
                alpha=0.3,
                label=label if ep_idx == 0 else None,   # only first gets legend
            )

    # Compute mean ± std across episodes in each bucket
    all_eps = [u for _, eps in buckets for u in eps]
    max_len = max(len(u) for u in all_eps) if all_eps else 0

    def compute_uncert_stats(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not arrays:
            return np.full(max_len, np.nan), np.full(max_len, np.nan)
        means, stds = [], []
        for t in range(max_len):
            vals = np.array([a[t] for a in arrays if len(a) > t and np.isfinite(a[t])])
            means.append(vals.mean() if vals.size else np.nan)
            stds.append(vals.std() if vals.size else np.nan)
        return np.array(means), np.array(stds)

    for label, episodes in buckets:
        mean, std = compute_uncert_stats([np.where(np.isneginf(u), np.nan, u) for u in episodes])
        plt.plot(np.arange(max_len), mean, color=colours[label])
        plt.fill_between(
            np.arange(max_len), mean - std, mean + std, color=colours[label], alpha=0.5
        )

    plt.xlabel("Action-Sequence Index")
    plt.ylabel("Uncertainty Score")
    plt.title(uncert_est_method.replace("_", " ").title())
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"uncertainty_scores_{uncert_est_method}.png",
                dpi=160, bbox_inches="tight")
    plt.close()

def rollout(
    env: gym.Env,
    policy: PreTrainedPolicy,
    seed: Optional[int],
) -> Dict[str, np.ndarray | bool]:
    device = get_device_from_parameters(policy)

    ep_uncertainties = []
    ep_frames = []
    success = False

    start_time = time.time() 

    # Reset the policy and environments.
    policy.reset()
    observation, _ = env.reset(seed=seed)

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
            uncertainty = policy.uncertainty_sampler.latest_uncertainties.detach().cpu().mean().item()
            
            ep_uncertainties.append(uncertainty)

        # Apply the next action
        observation, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
        ep_frames.append(env.render())

        if info is not None and "is_success" in info:
            success = bool(info["is_success"])

        # Stop early if environment terminates
        done = terminated or truncated
        if done:
            break

    logging.info(f"Finished episode in {time.time() - start_time:.1f}s")

    info = {
        "ep_uncertainties": ep_uncertainties,
        "ep_frames": ep_frames,
        "success": success,
    }

    return info

def load_failure_seeds(path: Path) -> list[int]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text())
    return [int(s) for s in data.get("failure_seeds", [])]

def choose_seed(failure_pool: list[int]) -> int:
    """
    50% chance to use a failure seed, otherwise return a fresh
    32-bit random seed. The chosen failure seed is removed from the
    pool so it isn't reused again in this run.
    """
    use_failure = bool(failure_pool) and random.random() < 0.5
    if use_failure:
        seed = random.choice(failure_pool)
        failure_pool.remove(seed)
        return seed

    return random.randrange(2**31 - 1)

@parser.wrap()
def main(cfg: EvalUncertaintyEstimationPipelineConfig): 
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    if cfg.policy.type != "flow_matching":
        raise ValueError(
            f"visualize_flow_matching.py only supports Flow Matching policies, "
            f"but got policy type '{cfg.policy.type}'."
        )
    
    id_failure_pool = load_failure_seeds(cfg.eval_uncert_est.id_failure_seeds_path)
    ood_failure_pool = load_failure_seeds(cfg.eval_uncert_est.ood_failure_seeds_path)

    n_episods = cfg.eval_uncert_est.n_episodes
    all_uncertainties = defaultdict(lambda: defaultdict(list))
    progbar = trange(
        n_episods,
        desc=f"Evaluating uncertainty estimation for {n_episods} episodes."
    )
    for episode in progbar:
        for uncert_est_method in cfg.eval_uncert_est.uncert_est_methods:
            logging.info("Loading policy")
            cfg.uncertainty_sampler.type = uncert_est_method
            device = get_safe_torch_device(cfg.policy.device, log=True)
            policy = make_policy(
                cfg.policy,
                env_cfg=cfg.env,
                uncertainty_sampler_cfg=cfg.uncertainty_sampler
            ).to(device)
            policy.eval()
            if uncert_est_method == "cross_likelihood_laplace":
                laplace_cfg = cfg.uncertainty_sampler.cross_likelihood_laplace_sampler
                laplace_path = make_laplace_path(
                    repo_id=cfg.dataset.repo_id,
                    scope=laplace_cfg.laplace_scope,
                    calib_fraction=laplace_cfg.calib_fraction,
                    batch_size=laplace_cfg.batch_size,
                )
                if not laplace_path.exists():
                    laplace_calib_loader = create_laplace_flow_matching_calib_loader(
                        cfg=cfg,
                        policy=policy,
                    )
                else:
                    laplace_calib_loader = None
            else:
                laplace_calib_loader = None
                laplace_path = None
            policy._init_uncertainty_sampler(
                laplace_calib_loader=laplace_calib_loader,
                laplace_path=laplace_path,
            )

            # ------------ ID Case ------------------
            logging.info(f"Creating ID environment.")
            seed = choose_seed(id_failure_pool)
            cfg.env.perturbation.enable = False
            id_env = make_single_env(cfg.env)
            id_ep_info = rollout(
                env=id_env,
                policy=policy,
                seed=seed
            )
            
            if id_ep_info["success"]:
                all_uncertainties[uncert_est_method]["id_success"].append(id_ep_info["ep_uncertainties"])
                id_success_output_dir = cfg.output_dir / uncert_est_method / "id_success"
                id_success_output_dir.mkdir(parents=True, exist_ok=True)
                write_video(
                    str(id_success_output_dir / f"rollout_ep{(episode + 1):03d}.mp4"),
                    np.stack(id_ep_info["ep_frames"], axis=0),
                    fps=id_env.metadata["render_fps"]
                )
            else:
                all_uncertainties[uncert_est_method]["id_failure"].append(id_ep_info["ep_uncertainties"])
                id_failure_output_dir = cfg.output_dir / uncert_est_method / "id_failure"
                id_failure_output_dir.mkdir(parents=True, exist_ok=True)
                write_video(
                    str(id_failure_output_dir / f"rollout_ep{(episode + 1):03d}.mp4"),
                    np.stack(id_ep_info["ep_frames"], axis=0),
                    fps=id_env.metadata["render_fps"]
                )
            # -----------------------------------------

            policy.reset()

            # ------------ OoD Case ------------------
            logging.info(f"Creating OoD environment.")
            seed = choose_seed(ood_failure_pool)
            ood_env = make_single_env(
                replace(cfg.env, perturbation=cfg.eval_uncert_est.perturbation_config),
            )
            ood_ep_info = rollout(
                env=ood_env,
                policy=policy,
                seed=seed
            )
                
            if ood_ep_info["success"]:
                all_uncertainties[uncert_est_method]["ood_success"].append(ood_ep_info["ep_uncertainties"])
                ood_success_output_dir = cfg.output_dir / uncert_est_method / "ood_success"
                ood_success_output_dir.mkdir(parents=True, exist_ok=True)
                write_video(
                    str(ood_success_output_dir / f"rollout_ep{(episode + 1):03d}.mp4"),
                    np.stack(ood_ep_info["ep_frames"], axis=0),
                    fps=ood_env.metadata["render_fps"]
                )
            else:
                all_uncertainties[uncert_est_method]["ood_failure"].append(ood_ep_info["ep_uncertainties"])
                ood_failure_output_dir = cfg.output_dir / uncert_est_method / "ood_failure"
                ood_failure_output_dir.mkdir(parents=True, exist_ok=True)
                write_video(
                    str(ood_failure_output_dir / f"rollout_ep{(episode + 1):03d}.mp4"),
                    np.stack(ood_ep_info["ep_frames"], axis=0),
                    fps=ood_env.metadata["render_fps"]
                )

            plot_all_uncertainties(
                uncert_est_method = uncert_est_method,
                id_success_uncertainties = all_uncertainties[uncert_est_method]["id_success"],
                id_failure_uncertainties = all_uncertainties[uncert_est_method]["id_failure"],
                ood_success_uncertainties = all_uncertainties[uncert_est_method]["ood_success"],
                ood_failure_uncertainties = all_uncertainties[uncert_est_method]["ood_failure"],
                output_dir = cfg.output_dir / uncert_est_method,
            )

            policy.cpu()
            del policy
            torch.cuda.empty_cache()

if __name__ == "__main__":
    init_logging()
    main()