#!/usr/bin/env python
"""
Evaluate a Flow Matching policy's uncertainty estimates for in-distribution and out-of-distribution
scenarios.

Runs rollouts with multiple uncertainty estimation methods, records per-step uncertainty scores,
saves rollout videos, and generates comparison plots for each method.

Usage example:

You want to plot uncertainty estimation scores for the Action Sequence Likelihood and the Composed 
Action Sequence Likelihood methods for 10 episodes:

```
python lerobot/scripts/eval_uncertainty_estimation.py \
    --policy.path=outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht \
    --eval_uncert_est.n_episodes=10 \
    --eval_uncert_est.uncert_est_methods='["likelihood", "composed_sequence"]'
```
"""
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.flow_matching.uncertainty.laplace_utils import (
    create_laplace_flow_matching_calib_loader,
    make_laplace_path,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.configs.eval_uncertainty_estimation import EvalUncertaintyEstimationPipelineConfig


def plot_id_ood_uncertainties(
    uncert_est_method: str,
    id_uncertainties: list[np.ndarray],
    ood_uncertainties: list[np.ndarray],
    output_dir: Path,
    plot_individual: bool,
):
    """
    Draw per-episode uncertainties and mean ± std bands for ID and OoD scenarios.
    """
    # Consistent colours
    colours = {"ID": "C0", "OoD": "C1"}
    plt.figure()
    
    # Plot every episode as a faint line
    if plot_id_ood_uncertainties:
        for ep_idx, uncert in enumerate(id_uncertainties):
            uncert = np.where(np.isneginf(uncert), np.nan, uncert)
            plt.plot(
                np.arange(len(uncert)),
                uncert,
                color=colours["ID"],
                alpha=0.3,
                label="In-Distribution" if ep_idx == 0 else None
            )
        for ep_idx, uncert in enumerate(ood_uncertainties):
            uncert = np.where(np.isneginf(uncert), np.nan, uncert)
            plt.plot(
                np.arange(len(uncert)),
                uncert,
                color=colours["OoD"],
                alpha=0.3,
                label="Out-of-Distribution" if ep_idx == 0 else None
            )

    # Compute mean ± std across episodes in each bucket
    all_eps = [u for eps in [id_uncertainties, ood_uncertainties] for u in eps]
    max_len = max(len(u) for u in all_eps) if all_eps else 0
    
    def compute_uncert_stats(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not arrays:
            return np.full(max_len, np.nan), np.full(max_len, np.nan)
        mean, std = [], []
        for t in range(max_len):
            vals = np.array([a[t] for a in arrays if len(a) > t and np.isfinite(a[t])])
            mean.append(vals.mean() if vals.size else np.nan)
            std.append(vals.std() if vals.size else np.nan)
        return np.asarray(mean), np.asarray(std)

    id_mean, id_std = compute_uncert_stats(id_uncertainties)
    ood_mean, ood_std = compute_uncert_stats(ood_uncertainties)

    for label, mean, std in [("ID", id_mean, id_std), ("OoD", ood_mean, ood_std)]:
        plt.plot(np.arange(len(mean)), mean, color=colours[label])
        plt.fill_between(
            np.arange(len(mean)), mean - std, mean + std, color=colours[label], alpha=0.5
        )

    plt.xlabel("Action-Sequence Index")
    plt.ylabel("Uncertainty Score")
    plt.title(uncert_est_method.replace("_", " ").title())
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_individual" if plot_individual else ""
    plt.savefig(
        output_dir / f"uncertainty_scores_{uncert_est_method}{suffix}.png",
        dpi=160, bbox_inches="tight"
    )
    plt.close()


def plot_all_uncertainties(
    uncert_est_method: str,
    scoring_metric: str,
    id_success_uncertainties: list[np.ndarray],
    id_failure_uncertainties: list[np.ndarray],
    ood_success_uncertainties: list[np.ndarray],
    ood_failure_uncertainties: list[np.ndarray],
    output_dir: Path,
    plot_individual: bool,
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
    buckets = [
        ("ID success", id_success_uncertainties),
        ("ID failure", id_failure_uncertainties),
        ("OoD success", ood_success_uncertainties),
        ("OoD failure", ood_failure_uncertainties),
    ]
    plt.figure()
    
    # Plot every episode as a faint line
    if plot_individual:
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
        plt.plot(
            np.arange(max_len),
            mean,
            color=colours[label],
            label=None if plot_individual else label
        )
        plt.fill_between(
            np.arange(max_len), mean - std, mean + std, color=colours[label], alpha=0.5
        )

    # Presentation: fontsize=18, labelpad=4
    plt.xlabel("Rollout Step")
    
    if scoring_metric == "mode_distance":
        y_label = "Mode Distance Score"
    elif scoring_metric == "inter_vel_diff":
        y_label = "Velocity Diff. Score"
    elif scoring_metric == "likelihood":
        y_label = "Neg. Log-Likelihood"
    else:
        y_label = "Uncertainty Score"
    # Presentation: fontsize=18, labelpad=4
    plt.ylabel(ylabel=y_label)
    plt.title(uncert_est_method.replace("_", " ").title())
    plt.legend()
    # Presentation
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend(fontsize=15)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_individual" if plot_individual else ""
    # Presentation dpi=600
    plt.savefig(
        output_dir / f"uncertainty_scores_{uncert_est_method}{suffix}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()


def rollout(
    env: gym.Env,
    policy: PreTrainedPolicy,
    seed: Optional[int],
) -> Dict[str, np.ndarray | bool]:
    device = get_device_from_parameters(policy)

    # Initialize random number generator to deterministically select actions
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    ep_uncertainties = []
    if hasattr(env, "camera_names") and env.camera_names is not None:
        ep_frames: Dict[str, list[np.ndarray]] = {
            cam: [] for cam in env.camera_names
        }
    else:
        ep_frames: list[np.ndarray] = []
    success = False

    start_time = time.time() 

    # Reset the policy and environments.
    policy.reset()
    observation, _ = env.reset(seed=seed)

    if hasattr(env, "camera_names") and env.camera_names is not None:
        for camera in env.camera_names:
            ep_frames[camera].append(env.unwrapped.render(camera_name=camera))
    else:
        ep_frames.append(env.render())
    
    max_episode_steps = env.spec.max_episode_steps
    # max_episode_steps = 30
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

        # Decide whether a new sequence will be generated
        new_action_gen = len(policy._queues["action"]) == 0

        with torch.no_grad():
            action = policy.select_action(observation, generator)

        if new_action_gen:
            uncertainty = policy.uncertainty_sampler.latest_uncertainties.detach().cpu().mean().item()
            
            ep_uncertainties.append(uncertainty)

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


def choose_seed(failure_pool: list[int], rng: Optional[random.Random] = None) -> int:
    """
    50% chance to use a failure seed, otherwise return a fresh
    32-bit random seed. The chosen failure seed is removed from the
    pool so it isn't reused again in this run.
    """
    use_failure = bool(failure_pool) and rng.random() < 0.5
    if use_failure:
        seed = rng.choice(failure_pool)
        failure_pool.remove(seed)
        return seed

    return rng.randrange(2**31 - 1)


def save_episode_video(
    ep_frames: Union[List[np.ndarray], Dict[str, List[np.ndarray]]],
    out_root: Path,
    episode_idx: int,
    fps: int,
) -> None:
    ep_str = f"rollout_ep{episode_idx:03d}.mp4"

    if isinstance(ep_frames, list):
        out_root.mkdir(parents=True, exist_ok=True)
        write_video(
            str(out_root / ep_str),
            np.stack(ep_frames, axis=0),           # (T, H, W, C)
            fps=fps,
        )

    elif isinstance(ep_frames, dict):
        for cam, frames in ep_frames.items():
            cam_dir = out_root / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            write_video(
                str(cam_dir / ep_str),
                np.stack(frames, axis=0),
                fps=fps,
            )

    else:
        raise TypeError(f"ep_frames must be list or dict, got {type(ep_frames)}")


@parser.wrap()
def main(cfg: EvalUncertaintyEstimationPipelineConfig): 
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Random number generator to choose seeds for the single runs
    rng = random.Random(cfg.seed)
    
    if cfg.policy.type != "flow_matching":
        raise ValueError(
            f"eval_uncertainty_estimation.py only supports Flow Matching policies, "
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
        seed = choose_seed(ood_failure_pool, rng)
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
            laplace_calib_loader = None
            laplace_path = None
            if getattr(cfg.uncertainty_sampler.active_config, "scorer_type", None) == "laplace":
                laplace_path = make_laplace_path(
                    repo_id=cfg.dataset.repo_id,
                    scope=cfg.uncertainty_sampler.active_config.laplace_scope,
                    calib_fraction=cfg.uncertainty_sampler.active_config.calib_fraction,
                )
                if not laplace_path.exists():
                    laplace_calib_loader = create_laplace_flow_matching_calib_loader(
                        cfg=cfg,
                        policy=policy,
                        calib_fraction=cfg.uncertainty_sampler.active_config.calib_fraction,
                        batch_size=cfg.uncertainty_sampler.active_config.batch_size
                    )
            policy.init_uncertainty_sampler(
                laplace_calib_loader=laplace_calib_loader,
                laplace_path=laplace_path,
            )
            scoring_metric = getattr(policy.uncertainty_sampler, 'scoring_metric', None)

            # ------------ ID Case ------------------
            logging.info("Creating ID environment.")
            seed = choose_seed(id_failure_pool, rng)
            cfg.env.ood.enabled = False
            id_env = make_single_env(cfg.env, seed)
            id_ep_info = rollout(
                env=id_env,
                policy=policy,
                seed=seed
            )
            
            if id_ep_info["success"]:
                all_uncertainties[uncert_est_method]["id_success"].append(id_ep_info["ep_uncertainties"])
                save_episode_video(
                    ep_frames=id_ep_info["ep_frames"],
                    out_root=cfg.output_dir / uncert_est_method / "id_success",
                    episode_idx=episode + 1,
                    fps=id_env.metadata["render_fps"],
                )
            else:
                all_uncertainties[uncert_est_method]["id_failure"].append(id_ep_info["ep_uncertainties"])
                save_episode_video(
                    ep_frames=id_ep_info["ep_frames"],
                    out_root=cfg.output_dir / uncert_est_method / "id_failure",
                    episode_idx=episode + 1,
                    fps=id_env.metadata["render_fps"],
                )
            # -----------------------------------------

            policy.reset()

            # ------------ OoD Case ------------------
            logging.info("Creating OoD environment.")
            seed = choose_seed(ood_failure_pool, rng)
            cfg.env.ood.enabled = True
            ood_env = make_single_env(cfg.env, seed)
            ood_ep_info = rollout(
                env=ood_env,
                policy=policy,
                seed=seed
            )
                
            if ood_ep_info["success"]:
                all_uncertainties[uncert_est_method]["ood_success"].append(ood_ep_info["ep_uncertainties"])
                save_episode_video(
                    ep_frames=ood_ep_info["ep_frames"],
                    out_root=cfg.output_dir / uncert_est_method / "ood_success",
                    episode_idx=episode + 1,
                    fps=ood_env.metadata["render_fps"],
                )
            else:
                all_uncertainties[uncert_est_method]["ood_failure"].append(ood_ep_info["ep_uncertainties"])
                save_episode_video(
                    ep_frames=ood_ep_info["ep_frames"],
                    out_root=cfg.output_dir / uncert_est_method / "ood_failure",
                    episode_idx=episode + 1,
                    fps=ood_env.metadata["render_fps"],
                )

            if cfg.eval_uncert_est.collapse_success_failure:
                id_all = (all_uncertainties[uncert_est_method]["id_success"] +
                          all_uncertainties[uncert_est_method]["id_failure"])
                ood_all = (all_uncertainties[uncert_est_method]["ood_success"] +
                           all_uncertainties[uncert_est_method]["ood_failure"])

                plot_id_ood_uncertainties(
                    uncert_est_method=uncert_est_method,
                    id_uncertainties=id_all,
                    ood_uncertainties=ood_all,
                    output_dir=cfg.output_dir / uncert_est_method,
                    plot_individual=False,
                )
                plot_id_ood_uncertainties(
                    uncert_est_method=uncert_est_method,
                    id_uncertainties=id_all,
                    ood_uncertainties=ood_all,
                    output_dir=cfg.output_dir / uncert_est_method,
                    plot_individual=True,
                )
            else:
                plot_all_uncertainties(
                    uncert_est_method=uncert_est_method,
                    scoring_metric=scoring_metric,
                    id_success_uncertainties=all_uncertainties[uncert_est_method]["id_success"],
                    id_failure_uncertainties=all_uncertainties[uncert_est_method]["id_failure"],
                    ood_success_uncertainties=all_uncertainties[uncert_est_method]["ood_success"],
                    ood_failure_uncertainties=all_uncertainties[uncert_est_method]["ood_failure"],
                    output_dir=cfg.output_dir / uncert_est_method,
                    plot_individual=False
                )
                plot_all_uncertainties(
                    uncert_est_method=uncert_est_method,
                    scoring_metric=scoring_metric,
                    id_success_uncertainties=all_uncertainties[uncert_est_method]["id_success"],
                    id_failure_uncertainties=all_uncertainties[uncert_est_method]["id_failure"],
                    ood_success_uncertainties=all_uncertainties[uncert_est_method]["ood_success"],
                    ood_failure_uncertainties=all_uncertainties[uncert_est_method]["ood_failure"],
                    output_dir=cfg.output_dir / uncert_est_method,
                    plot_individual=True
                )

            policy.cpu()
            del policy
            torch.cuda.empty_cache()

if __name__ == "__main__":
    init_logging()
    main()