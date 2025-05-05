#!/usr/bin/env python
"""
Test flow matching log-likelihood computation of a target sample x_1.

Examples:
    - Estimate the log-likelihood computation of target sample x_1 for a flow
    matching policy trained on the Push-T dataset:
    ```
    local$ python lerobot/scripts/test_flow_matching_likelihood_computation.py \
        -r lerobot/pusht \
        -p outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model
    ```
"""
import argparse
import torch

from torch.distributions import Independent, Normal

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingPolicy
from lerobot.common.policies.flow_matching.ode_solver import ODESolver
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig

def test_log_likelihood_computation(
    ds_repo_id: str,
    pretrained_flow_matching_path: str,
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
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
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
    global_cond = flow_matching_model._prepare_global_conditioning(batch)

    # Create a noise generator with a fixed seed for reproducibility
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)

    sample = flow_matching_model.conditional_sample(
        batch_size=batch_size,
        global_cond=global_cond,
        generator=generator,
    )

    print(f"Sample x_1: {sample}")

    ode_solver = ODESolver(velocity_model=flow_matching_model.unet)

    # Noise distribution is an isotropic gaussian
    horizon = flow_matching_policy.config.horizon
    action_dim = flow_matching_policy.config.action_feature.shape[0]
    gaussian_log_density = Independent(
        Normal(
            loc = torch.zeros(horizon, action_dim, device=device),
            scale = torch.ones(horizon, action_dim, device=device),
        ),
        reinterpreted_batch_ndims=2
    ).log_prob

    exact_divergence = False
    x_0, log_p_1_x_1 = ode_solver.compute_log_likelihood(
        x_1=sample,
        global_cond=global_cond,
        log_p_0=gaussian_log_density,
        step_size=flow_matching_policy.config.ode_step_size,
        method=flow_matching_policy.config.ode_solver_method,
        atol=flow_matching_policy.config.atol,
        rtol=flow_matching_policy.config.rtol,
        exact_divergence=exact_divergence
    )

    print("-------------------------------")
    print("Exact Divergence") if exact_divergence else print("Hutchinson")
    print("-------------------------------")
        
    print(f"Reconstructed x_0: {x_0}")
    print(f"Log-likelihood of x_1: {log_p_1_x_1}")

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
        help="Path to a pretrained flow‐matching policy checkpoint file.",
    )

    args = parser.parse_args()
    test_log_likelihood_computation(
        args.repo_id,
        args.pretrained_fm_path,
    )

if __name__ == "__main__":
    main()