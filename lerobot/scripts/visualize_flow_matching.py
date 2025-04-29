#!/usr/bin/env python
"""
Visualize flows produced by a trained flow matching policy to generate actions.

Example:
    - Visualize the flow field of a policy trained on the Push-T dataset:
    ```
    local$ python lerobot/scripts/visualize_flow_matching.py \
        --repo-id lerobot/pusht \
        --pretrained-fm-path outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model
    ```
"""

import argparse
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
        shuffle=False,
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

    # Encode image features and concatenate them all together along with the state vector.
    global_cond = flow_matching_model._prepare_global_conditioning(batch).squeeze(0)  # (B, global_cond_dim)

    flow_matching_visualizer = FlowMatchingVisualizer(
        config=flow_matching_policy.config,
        velocity_model=flow_matching_model.unet,
        global_cond=global_cond,
        action_dim_names=["Position x", "Position y"]
    )
    flow_matching_visualizer.visualize_flows()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )

    parser.add_argument(
        "--pretrained-fm-path",
        type=str,
        required=True,
        help="Path to a pretrained flow‐matching policy checkpoint file.",
    )

    args = parser.parse_args()
    args = parser.parse_args()
    visualize_flow_matching_flows(
        args.repo_id,
        args.pretrained_fm_path,
    )

if __name__ == "__main__":
    main()