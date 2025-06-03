#!/usr/bin/env python
"""
Sample action sequences and estimate their uncertainties for a given
conditioning feature vector using a trained flow matching policy.

Examples:
    - Sample action sequences and estimate their uncertainties using the epsilon-ball
    expansion method, given a flow matching policy trained on the Push-T dataset :
    ```
    local$ python lerobot/scripts/estimate_flow_matching_uncertainty.py \
        -r lerobot/pusht \
        -p outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
        -m epsilon_ball

    - Sample action sequences and estimate their uncertainty using the their negative
    log-likelihoods, given a flow matching policy trained on the Push-T dataset.
    ```
    local$ python lerobot/scripts/estimate_flow_matching_uncertainty.py \
        -r lerobot/pusht \
        -p outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
        -m likelihood
    ```
"""
import argparse
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.flow_matching.estimate_uncertainty import EpsilonBallSampler, LikelihoodSampler
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig

def estimate_flow_matching_uncertainty(
    ds_repo_id: str,
    pretrained_flow_matching_path: str,
    method: str,
):
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
        if flow_matching_policy.config.n_obs_steps == 1:
            batch["observation.images"] = batch["observation.images"].unsqueeze(1)
    
    # Encode image features and concatenate them all together along with the state vector.
    global_cond = flow_matching_model._prepare_global_conditioning(batch).squeeze(0)  # (B, global_cond_dim)

    # Create a noise generator with a fixed seed for reproducibility
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)

    num_action_seq_samples = 3
    if method == "epsilon_ball":
        fm_uncertainty_sampler = EpsilonBallSampler(
            config=flow_matching_policy.config,
            velocity_model=flow_matching_model.unet,
            num_action_seq_samples=num_action_seq_samples,
            num_eps_ball_samples=100,
            generator=generator,
        )
    elif method == "likelihood":
        fm_uncertainty_sampler = LikelihoodSampler(
            config=flow_matching_policy.config,
            velocity_model=flow_matching_model.unet,
            num_action_seq_samples=num_action_seq_samples,

            generator=generator,
        )

    action_seqs, uncertainty_scores = fm_uncertainty_sampler.conditional_sample_with_uncertainty(global_cond=global_cond)
    
    print("Sampled action sequences:")
    print(action_seqs)
    print("Uncertainty scores:")
    print(uncertainty_scores)

def main():
    parser = argparse.ArgumentParser()

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
        help="Path to a pretrained flow matching policy checkpoint file.",
    )

    parser.add_argument(
        "-m", "--method",
        type=str,
        required=True,
        choices=["epsilon_ball", "likelihood"],
        help="Method to estimate uncertainty."
    )

    args = parser.parse_args()
    estimate_flow_matching_uncertainty(
        args.repo_id,
        args.pretrained_fm_path,
        args.method,
    )

if __name__ == "__main__":
    main()