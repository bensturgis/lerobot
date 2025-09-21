import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from torch import Tensor, nn
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.flow_matching.conditional_probability_path import OTCondProbPath
from lerobot.common.policies.flow_matching.modelling_flow_matching import (
    FlowMatchingModel,
    FlowMatchingPolicy,
)
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig


@dataclass(frozen=True)
class FlowMatchingInput:
    """
    Container for one batch of inputs to a flow matching model including RGB encoder.
    """
    # Interpolated trajectory based on some random noise sample and a target action
    # using an optimal transport conditional probability path.
    interp_traj: torch.Tensor # Shape: (batch_size, horizon, action_dim)
    # Time step along the flow path.
    time: torch.Tensor # Shape: (batch_size,)
    # Input observations of the environment.
    observation: Dict[str, torch.Tensor] # List of length batch_size

    def to(self, *args, **kwargs) -> "FlowMatchingInput":
        """
        Return a copy of this FlowMatchingInput with all contained tensors moved
        or cast.
        """
        return FlowMatchingInput(
            interp_traj = self.interp_traj.to(*args, **kwargs),
            time = self.time.to(*args, **kwargs),
            observation = {k: v.to(*args, **kwargs) for k, v in self.observation.items()}
        )

    def detach(self) -> "FlowMatchingInput":
        return FlowMatchingInput(
            interp_traj = self.interp_traj.detach(),
            time = self.time.detach(),
            observation = {k: v.detach() for k, v in self.observation.items()}
        )

    def cpu(self) -> "FlowMatchingInput":
        return self.to("cpu")

def make_laplace_collate(
    policy_cfg: PreTrainedConfig,
    policy: FlowMatchingPolicy,
):
    """
    Factory that builds a dataloader collate function tailored for laplace-torch
    calibration of a flow matching policy.

    The returned converts a list of raw samples coming from the training dataset
    into the pair (FlowMatchingInput, target_velocity) expected by to fit a Laplace
    posterior using laplace-torch.

    Args:
        policy_cfg: Policy config.
        policy: Trained flow matching policy providing normalization.
    """
    # Check device is available
    device = get_safe_torch_device(policy_cfg.device)

    # Re-use the same OT path object for every batch
    ot_cond_prob_path = OTCondProbPath()

    def laplace_collate(raw_batch):
        batch = default_collate(raw_batch)

        # Move all tensors in the batch to the correct device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)

        with torch.no_grad():
            batch = policy.normalize_inputs(batch)

            if policy_cfg.image_features:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch["observation.images"] = torch.stack(
                    [batch[key] for key in policy_cfg.image_features], dim=-4
                )
                if policy_cfg.n_obs_steps == 1:
                    batch["observation.images"] = batch["observation.images"].unsqueeze(1)
            batch = policy.normalize_targets(batch)

            # Get ground-truth trajectory (x_1)
            trajectory = batch["action"]
            batch_size = trajectory.shape[0]
            
            # Sample a random time for each item in the batch.
            times = torch.rand(
                size=(batch_size,),
                device=device,
            )
            # Sample from conditional probability path and vector field.
            interpolated_trajectory = ot_cond_prob_path.sample(
                x_1=trajectory,
                t=times[:, None, None],
            )
            target_vel = ot_cond_prob_path.velocity(
                x_t=interpolated_trajectory,
                x_1=trajectory,
                t=times[:, None, None],
            )

        # Create observation only dictionary
        observation = {
            key: value.cpu()
            for key, value in batch.items()
            if key.startswith("observation.")
        }

        # Package inputs and targets exactly as laplace-torch expects
        input_batch = FlowMatchingInput(
            interp_traj = interpolated_trajectory,
            time = times,
            observation = observation,
        )
        target_batch = target_vel.flatten(start_dim=1)

        return input_batch, target_batch

    return laplace_collate

@torch.no_grad()
def create_laplace_flow_matching_calib_loader(
    dataset_cfg: DatasetConfig,
    policy_cfg: PreTrainedConfig,
    policy: FlowMatchingPolicy,
    calib_fraction: float,
    batch_size: int,
) -> DataLoader:
    """
    Build a data loader for Laplace approximation calibration of a flow matching model
    including RGB encoder.

    Args:
        dataset_cfg: Dataset construction parameters.
        policy_cfg: Policy configuration.
        policy: A trained flow matching policy.
        calib_fraction: Fraction of the full dataset to reserve for calibration
            (between 0 and 1).
        batch_size: Number of samples per batch in the returned DataLoader.
            
    Returns:
        A data loader over a small calibration set yielding batches of the form
        ((interpolated_trajectory, time_step, observation), target_velocity).
        This corresponds to the input and target output of the flow matching model
        including RGB encoder. This data loader matches the 'laplace.LaPlace.fit()' method.
    """
    # Extract a subset of the full train dataset for calibration
    train_dataset = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_cfg)
    num_train_samples = len(train_dataset)
    num_calib_samples = int(calib_fraction * num_train_samples)
    calib_indices = torch.randperm(num_train_samples)[:num_calib_samples].tolist()
    calib_subset = torch.utils.data.Subset(train_dataset, calib_indices)

    calib_loader = torch.utils.data.DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=True,                  # shuffle within the subset
        num_workers=0,
        collate_fn=make_laplace_collate(policy_cfg=policy_cfg, policy=policy)
    )

    return calib_loader

def draw_laplace_flow_matching_model(
    laplace_posterior: BaseLaplace, 
    flow_matching_model: nn.Module,
    generator: Optional[torch.Generator] = None,
) -> FlowMatchingModel:
    """
    Sample a single set of weights from a fitted Laplace posterior and
    insert it into a copy of the MAP flow matching model.

    Args:
        laplace_posterior: A fitted Laplace posterior.
        flow_matching_model: The original MAP flow matching model. It is not modified,
            but its architecture is only used as a template for the copy that will receive
            the sampled weights. .
        generator: Generator for reproducible sampling.

    Returns:
        A new flow-matching model in which all parameters included in the Laplace posterior
        have been replaced by a single Monte Carlo sample drawn from the posterior distribution.
    """
    # Draw weights from the Laplace posterior
    laplace_model_weights = laplace_posterior.sample(
        n_samples=1,
        generator=generator
    ).squeeze(0)

    # Copy the MAP model so we never mutate the original
    laplace_flow_matching_model = copy.deepcopy(flow_matching_model)

    # Collect the parameters that were in the posterior
    target_params = [p for p in laplace_flow_matching_model.parameters() if p.requires_grad]

    # Consistency check to avoid silent weight mis-alignment
    n_expected = sum(p.numel() for p in target_params)
    if laplace_model_weights.numel() != n_expected:
        raise RuntimeError(
            f"[Laplace] Sample size mismatch: drew {laplace_model_weights.numel()} "
            f"weights but found {n_expected} trainable parameters in the copy."
        )

    # Write sampled parameters into the copied model (in-place assignment)
    vector_to_parameters(laplace_model_weights, target_params)
    
    # Move the model to the same device as sampled weights and switch to inference mode
    laplace_flow_matching_model = laplace_flow_matching_model.to(laplace_model_weights.device)
    laplace_flow_matching_model.eval()

    return laplace_flow_matching_model

class FlowMatchingModelWrapper(nn.Module):
    """
    Wraps a Flow Matching model including RGB encoder so that laplace-torch can call it
    with a single FlowMatchingInput object.
    """
    def __init__(self, flow_matching_model: nn.Module) -> None:
        super().__init__()
        self.base_model = flow_matching_model

    def forward(self, input: FlowMatchingInput):
        # Unpack FlowMatchingInput attributes and forward them to the velocity model
        output = self.base_model(input.interp_traj, input.time, input.observation)
        return output.flatten(start_dim=1) 

class PointwiseConv1dToLinear(nn.Module):
    """
    Adapter that replaces a pretrained 1x1 Conv1D layer with an equivalent nn.Linear,
    so that Laplace-torch can recognize it as a linear leaf module. The functional
    output is identical to the original convolution.
    """
    def __init__(self, conv_layer: nn.Conv1d):
        super().__init__()
        # Number of output channels for reshaping
        self.out_channels = conv_layer.out_channels
        # Create a Linear layer matching the conv_layer parameters
        self.linear_layer = nn.Linear(
            in_features=conv_layer.in_channels,
            out_features=conv_layer.out_channels,
            bias=(conv_layer.bias is not None),
            device=conv_layer.weight.device,
            dtype=conv_layer.weight.dtype,
        )
        # Copy pretrained conv weights and bias
        with torch.no_grad():
            # conv_layer.weight shape: (C_out, C_in, 1)
            self.linear_layer.weight.copy_(conv_layer.weight.squeeze(-1))
            if conv_layer.bias is not None:
                self.linear_layer.bias.copy_(conv_layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass that applies the equivalent linear transformation at each time step.

        Args:
            inputs: Inputs to the final 1x1 Conv1D layer. Shape (batch_size, C_in, horizon).

        Returns:
           Outputs of the final 1x1 Conv1D layer. Shape (batch_size, action_dim, horizon).
        """
        # Inputs Shape: (batch_size, C_in, horizon)
        batch_size, _, horizon = inputs.shape
        # Flatten horizon dimension into batch dimension: (bach_size*horizon, C_in)
        flat_inputs = inputs.permute(0, 2, 1).reshape(batch_size * horizon, -1)
        # Apply linear layer: (bach_size*horizon, action_dim)
        flat_outputs = self.linear_layer(flat_inputs)
        # Restore shape: (bach_size*horizon, action_dim, horizon)
        outputs = flat_outputs.view(batch_size, horizon, self.out_channels).permute(0, 2, 1)
        return outputs
    
REPO_DIR_MAP: Dict[str, str] = {
    "lerobot/pusht": "pusht",
    "lerobot/aloha_sim_transfer_cube_human": "aloha_transfer",
    "lerobot/libero_spatial_one_bowl": "libero",
}

def resolve_repo_dir(repo_id: str) -> str:
    try:
        return REPO_DIR_MAP[repo_id]
    except KeyError:
        raise ValueError(
            f"Unknown repo_id {repo_id}. Expected one of {list(REPO_DIR_MAP)}"
        )

def make_laplace_path(
    repo_id: str,
    scope: str,
    calib_fraction: float,
) -> Path:
    """
    Build (and create) the on-disk path where we save/load a Laplace posterior.
    """
    base_dir = Path("outputs") / "laplace_posterior"
    subdir = resolve_repo_dir(repo_id)
    out_dir = base_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    pct = int(calib_fraction * 100)
    fname = f"laplace_{scope}_frac{pct}pct.bin"
    return out_dir / fname

def get_laplace_posterior(
    laplace_scope: Literal["velocity_last", "rgb_last", "both"],
    flow_matching_model: nn.Module,
    laplace_calib_loader: Optional[torch.utils.data.DataLoader],
    laplace_path: str,
) -> Tuple[Laplace, List[str]]:
    """
    Build (or load) a diagonal Laplace posterior for a sub-network of a
    flow matching model.
    
    Args:
        laplace_scope: Which part of the model to approximate
            - "velocity_last": final layer of the velocity model
            - "rgb_last": final layer of the RGB encoder
            - "both": apply Laplace on both layers jointly
        flow_matching_model: The full flow matching model including velocity and RGB encoder.
        laplace_calib_loader: DataLoader providing samples for fitting the Laplace
                approximation.
        laplace_path: Path to save or load the Laplace posterior.

    Returns:
        - A fitted (or loaded) diagonal Laplace posterior over the specified
        sub-network weights.
        - Names of the modules that were selected (i.e., those with `requires_grad=True`)
        for the Laplace approximation.
    """
    # Select target modules for Laplace approximation
    laplace_approx_targets: list[str] = []
    if laplace_scope in ["velocity_last", "both"]:
        flow_matching_model.unet.final_conv[1] = PointwiseConv1dToLinear(
            flow_matching_model.unet.final_conv[1]
        )
        laplace_approx_targets.append("unet.final_conv.1.linear_layer")

    if laplace_scope in ["rgb_last", "both"]:
        laplace_approx_targets.append("rgb_encoder.out")

    if not laplace_approx_targets:
        raise ValueError(
            f"Unknown laplace_scope={laplace_scope}. "
            "Choose from ['velocity_last', 'rgb_last', 'both']"
        )

    # la_subnetwork_mask = ModuleNameSubnetMask(
    #     flow_matching_model,
    #     module_names=laplace_approx_targets
    # )
    # la_subnetwork_mask.select()
    # la_subnetwork_indices = la_subnetwork_mask.indices.cpu()

    # Freeze all parameters
    for p in flow_matching_model.parameters():
        p.requires_grad_(False)

    # Unfreeze only the selected subnetwork for Laplace fitting
    for name, module in flow_matching_model.named_modules():
        if name in laplace_approx_targets:
            for p in module.parameters():
                p.requires_grad_(True)

    # Wrap the flow matching model so it takes inputs and generates outputs
    # compatible with Laplace
    flow_matching_model.eval()
    wrapped_flow_matching_model = FlowMatchingModelWrapper(
        flow_matching_model
    )

    laplace_posterior = Laplace(
        wrapped_flow_matching_model,
        likelihood="regression",
        subset_of_weights="all",  # uses only params with requires_grad=True
        hessian_structure="diag",
    )

    if laplace_path.exists():
        logging.info(f"Loading Laplace posterior from {laplace_path}")
        laplace_posterior.load_state_dict(torch.load(laplace_path))
    else:
        logging.info("Fitting new Laplace posterior.")
        if laplace_calib_loader is None:
            raise ValueError("Calibration loader is required to fit Laplace.")
        laplace_posterior.fit(laplace_calib_loader)

        logging.info(f"Save Laplace posterior to {laplace_path}")
        torch.save(laplace_posterior.state_dict(), laplace_path)

    return laplace_posterior

def build_laplace_posterior_artifact(
    laplace_scope: str,
    calib_fraction: float,
    batch_size: int,
    dataset_cfg: DatasetConfig,
    policy_cfg: PreTrainedConfig,
    policy: FlowMatchingPolicy,
) -> Laplace:
    """
    Construct or load a Laplace posterior for the given flow matching model.

    Builds a calibration DataLoader if needed and fits a new posterior,
    otherwise loads an existing one from disk.

    Returns:
        A fitted or loaded Laplace posterior.
    """
    laplace_path = make_laplace_path(
        repo_id=dataset_cfg.repo_id,
        scope=laplace_scope,
        calib_fraction=calib_fraction,
    )
    calib_loader = None
    if not laplace_path.exists():
        calib_loader = create_laplace_flow_matching_calib_loader(
            dataset_cfg=dataset_cfg,
            policy_cfg=policy_cfg,
            policy=policy,
            calib_fraction=calib_fraction,
            batch_size=batch_size,
        )
    return get_laplace_posterior(
        laplace_scope=laplace_scope,
        flow_matching_model=policy.flow_matching,
        laplace_calib_loader=calib_loader,
        laplace_path=laplace_path,
    )