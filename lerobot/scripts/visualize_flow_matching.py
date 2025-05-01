#!/usr/bin/env python
"""
Visualize flows produced by a trained flow matching policy to generate actions.

Examples:
    - Visualize the flow field of a policy trained on the Push-T dataset:
    ```
    local$ python lerobot/scripts/visualize_flow_matching.py \
        -t flows \
        -r lerobot/pusht \
        -p outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model
    ```

    - Visualize vector field for a policy trained on Push-T dataset with a horizon of 1:
    local$ python lerobot/scripts/visualize_flow_matching.py \
        -t vector_field \
        -r lerobot/pusht \
        -p outputs/train/flow_matching_pusht_single_action/checkpoints/last/pretrained_model
"""

import argparse
import numpy as np
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy
from lerobot.common.policies.flow_matching.visualization_utils import FlowMatchingVisualizer
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig


def visualize_flow_matching_flows(ds_repo_id, pretrained_flow_matching_path):
    # Initialize flow matching visualizer using pretrained flow matching policy.
    flow_matching_policy = FlowMatchingPolicy.from_pretrained(pretrained_flow_matching_path)
    flow_matching_model = flow_matching_policy.flow_matching

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=ds_repo_id, episodes=[0]),
        policy=flow_matching_policy.config,
    )
    dataset = make_dataset(train_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        pin_memory=device != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)
    batch = next(dl_iter)

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)

    batch = flow_matching_policy.normalize_inputs(batch)
    if flow_matching_policy.config.image_features:
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch["observation.images"] = torch.stack(
            [batch[key] for key in flow_matching_policy.config.image_features], dim=-4
        )
        if flow_matching_policy.config.n_obs_steps == 1:
            batch["observation.images"] = batch["observation.images"].unsqueeze(1)

    # Encode image features and concatenate them all together along with the state vector.
    global_cond = flow_matching_model._prepare_global_conditioning(batch).squeeze(0)  # (B, global_cond_dim)

    flow_matching_visualizer = FlowMatchingVisualizer(
        config=flow_matching_policy.config,
        velocity_model=flow_matching_model.unet,
        global_cond=global_cond,
        action_dim_names=["Position x", "Position y"]
    )
    flow_matching_visualizer.visualize_flows()

def visualize_flow_matching_vector_field(ds_repo_id, pretrained_flow_matching_path):
    flow_matching_policy = FlowMatchingPolicy.from_pretrained(pretrained_flow_matching_path)
    flow_matching_model = flow_matching_policy.flow_matching

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=ds_repo_id, episodes=[0]),
        policy=flow_matching_policy.config,
    )
    dataset = make_dataset(train_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        pin_memory=device != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)
    batch = next(dl_iter)

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)

    batch = flow_matching_policy.normalize_inputs(batch)
    if flow_matching_policy.config.image_features:
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch["observation.images"] = torch.stack(
            [batch[key] for key in flow_matching_policy.config.image_features], dim=-4
        )
        if flow_matching_policy.config.n_obs_steps == 1:
            batch["observation.images"] = batch["observation.images"].unsqueeze(1)

    # Encode image features and concatenate them all together along with the state vector.
    global_cond = flow_matching_model._prepare_global_conditioning(batch).squeeze(0)  # (B, global_cond_dim)

    flow_matching_visualizer = FlowMatchingVisualizer(
        config=flow_matching_policy.config,
        velocity_model=flow_matching_model.unet,
        global_cond=global_cond,
        action_dim_names=["Position x", "Position y"]
    )

    # min_action = dataset.stats["action"]["min"]
    # max_action = dataset.stats["action"]["max"]

    min_action = np.array([-1, -1])
    max_action = np.array([1, 1])

    flow_matching_visualizer.visualize_vector_field(
        min_action=min_action,
        max_action=max_action,
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--vis_type",
        type=str,
        required=True,
        choices=["flows", "vector_field"],
        help=(
            "Type of visualization to generate:\n"
            "flows: per-step flow matching visualizations.\n"
            "vector_field: overall 2D action vector field (requires action_dim=2 and horizon=1)."
        ),
    )
    parser.add_argument(
        "-r", "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )

    parser.add_argument(
        "-p", "--pretrained-fm-path",
        type=str,
        required=True,
        help="Path to a pretrained flow‐matching policy checkpoint file.",
    )

    args = parser.parse_args()
    if args.vis_type == "flows":
        visualize_flow_matching_flows(
            args.repo_id,
            args.pretrained_fm_path,
        )
    elif args.vis_type == "vector_field":
        visualize_flow_matching_vector_field(
            args.repo_id,
            args.pretrained_fm_path,
        )

if __name__ == "__main__":
    main()