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

from lerobot.common.datasets.factory import make_dataset, make_train_val_split
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


@torch.no_grad()
def compute_val_loss(
    policy: PreTrainedPolicy,
    val_loader: torch.utils.data.DataLoader,
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
            # Move all tensors in the batch to the correct device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)

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
    full_dataset = make_dataset(cfg)
    if cfg.enable_val_loss:
        train_val_split = make_train_val_split(full_dataset, cfg)
        train_dataset, val_dataset = train_val_split.train_dataset, train_val_split.val_dataset
        train_ep_ids = train_val_split.train_ep_ids
    else:
        train_dataset = full_dataset
        val_dataset = None
        train_ep_ids = None

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=full_dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{full_dataset.num_frames=} ({format_big_number(full_dataset.num_frames)})")
    logging.info(f"{full_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            full_dataset.episode_data_index,
            episode_indices_to_use=train_ep_ids,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(train_loader)

    if cfg.enable_val_loss:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
    if cfg.val_freq is None:
        val_freq = math.ceil(len(train_dataset) / cfg.batch_size)
    else:
        val_freq = cfg.val_freq

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
        num_frames   = len(train_dataset) if cfg.enable_val_loss else train_dataset.num_frames,
        num_episodes = len(train_ep_ids) if cfg.enable_val_loss else train_dataset.num_episodes,
        metrics      = train_metrics,
        initial_step = step,
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

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
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
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
                eval_info = eval_policy(
                    cfg.env,
                    policy,
                    cfg.eval.n_episodes,
                    cfg.eval.batch_size,
                    cfg.eval.use_async_envs,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=10,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.eval.batch_size,
                cfg.eval.n_episodes,
                cfg.eval.n_episodes,
                eval_metrics,
                initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    logging.info("End of training")

if __name__ == "__main__":
    init_logging()
    train()
