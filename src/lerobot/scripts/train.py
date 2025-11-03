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
import math
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.compute_stats import compute_stats_for_episodes
from lerobot.datasets.factory import make_dataset, make_train_val_split
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle, filter_libero_episodes
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.processor import PolicyProcessorPipeline
from lerobot.scripts.eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


@torch.no_grad()
def compute_val_loss(
    policy: PreTrainedPolicy,
    val_loader: torch.utils.data.DataLoader,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """
    Iterate once over val_loader and return the mean validation loss.

    Args:
        policy: The policy to evaluate.
        val_loader: Dataloader that yields validation batches.
        device: Target device.
        use_amp: Enable AMP evaluation if the model was trained with AMP.

    Returns:
        Mean loss across the entire validation set.
    """
    loss_meter = AverageMeter("val_loss", fmt=":.3f")
    policy.eval()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        for batch in val_loader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            batch_size = batch[next(iter(batch))].size(0)
            loss_meter.update(loss.item(), n=batch_size)

    # Restore the training mode
    policy.train()
    return loss_meter.avg


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. It also handles mixed-precision training via a GradScaler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        grad_scaler: The GradScaler for automatic mixed-precision training.
        lr_scheduler: An optional learning rate scheduler.
        use_amp: A boolean indicating whether to use automatic mixed precision.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
    """
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    full_dataset = make_dataset(
        dataset_cfg=cfg.dataset,
        policy_cfg=cfg.policy,
        num_workers=cfg.num_workers,
    )

    all_episode_ids = list(full_dataset.meta.episodes["episode_index"])
    if cfg.dataset.repo_id == "HuggingFaceVLA/libero" and cfg.dataset.libero_tasks is not None:
        episode_ids_to_use = filter_libero_episodes(
            dataset=full_dataset,
            tasks_to_use=cfg.dataset.libero_tasks,
        )
    else:
        episode_ids_to_use = all_episode_ids

    if cfg.enable_val_loss:
        train_episode_ids, val_episode_ids = make_train_val_split(
            episode_ids=episode_ids_to_use,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
    else:
        train_episode_ids = episode_ids_to_use
        val_episode_ids = []

    # Create dataloader for offline training
    train_sampler = None
    if train_episode_ids != all_episode_ids or getattr(cfg.policy, "drop_n_last_frames", 0) > 0:
        if cfg.dataset.streaming:
            raise RuntimeError(
                "Episode filtering or frame dropping requires an episode aware sampler, "
                "which is disallowed in streaming mode."
            )
        train_sampler = EpisodeAwareSampler(
            dataset_from_indices=full_dataset.meta.episodes["dataset_from_index"],
            dataset_to_indices=full_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=sorted(train_episode_ids),
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=True,
        )

    train_shuffle = (train_sampler is None) and (not cfg.dataset.streaming)
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )

    # Effective sizes for logging and trackers
    effective_train_frames = len(train_sampler) if train_sampler is not None else len(full_dataset)
    effective_train_eps = len(train_episode_ids)

    val_loader = None
    val_sampler = None
    if cfg.enable_val_loss and len(val_episode_ids) > 0:
        val_sampler = EpisodeAwareSampler(
            dataset_from_indices=full_dataset.meta.episodes["dataset_from_index"],
            dataset_to_indices=full_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=sorted(val_episode_ids),
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        effective_val_frames = len(val_sampler)
        effective_val_eps = len(val_episode_ids)
    else:
        effective_val_frames = 0
        effective_val_eps    = 0

    if cfg.val_freq is None:
        val_freq = math.ceil(effective_train_frames / cfg.batch_size) if val_loader is not None else float("inf")
    else:
        val_freq = cfg.val_freq

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"Total number of episodes: {full_dataset.num_episodes} and frames: {full_dataset.num_frames} ({format_big_number(len(full_dataset))})")
    if cfg.dataset.repo_id == "HuggingFaceVLA/libero" and cfg.dataset.libero_tasks is not None:
        logging.info(f"Number of selected episodes: {len(episode_ids_to_use)} and frames: {effective_train_frames + effective_val_frames} ({format_big_number(effective_train_frames + effective_val_frames)})")
    logging.info(f"Number of train episodes: {effective_train_eps} and frames: {effective_train_frames} ({format_big_number(effective_train_frames)})")
    logging.info(f"Number of val episodes: {effective_val_eps} and frames: {effective_val_frames} ({format_big_number(effective_val_frames)})")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=full_dataset.meta,
    )

    if cfg.policy.pretrained_path and cfg.reinitialize_selected_layers:
        logging.info("Reinitialize selected layers")
        policy.reinitialize_selected_layers()

    if train_episode_ids != all_episode_ids:
        logging.info("Extracting dataset stats from training episodes")
        training_stats = compute_stats_for_episodes(full_dataset.meta.root, train_episode_ids)
    else:
        training_stats = full_dataset.meta.stats

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = training_stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": training_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": training_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dl_iter = cycle(train_loader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        batch_size   = cfg.batch_size,
        num_frames   = effective_train_frames,
        num_episodes = effective_train_eps,
        metrics      = train_metrics,
        initial_step = step,
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_val_step = step > 0 and step % val_freq == 0
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if cfg.enable_val_loss and is_val_step:
            val_loss = compute_val_loss(
                policy=policy,
                val_loader=val_loader,
                preprocessor=preprocessor,
                device=device,
                use_amp=cfg.policy.use_amp
            )
            logging.info(f"step {step}: Validation loss: {val_loss:.3f}")
            if wandb_logger:
                wandb_logger.log_dict({"loss": val_loss}, step, mode="val")

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler, preprocessor, postprocessor
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy_all(
                    envs=eval_env,  # dict[suite][task_id] -> vec_env
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=10,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            # overall metrics (suite-agnostic)
            aggregated = eval_info["overall"]

            # optional: per-suite logging
            for suite, suite_info in eval_info.items():
                logging.info("Suite %s aggregated: %s", suite, suite_info)

            # meters/tracker
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size,
                effective_train_frames,
                effective_train_eps,
                eval_metrics,
                initial_step=step
            )
            eval_tracker.eval_s = aggregated.pop("eval_s")
            eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
            eval_tracker.pc_success = aggregated.pop("pc_success")
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

    if eval_env:
        close_envs(eval_env)
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)
        preprocessor.push_to_hub(cfg.policy.repo_id)
        postprocessor.push_to_hub(cfg.policy.repo_id)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
