import copy
import logging
import torch

from dataclasses import dataclass
from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from pathlib import Path
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils import vector_to_parameters
from tqdm.auto import trange
from typing import Dict, List, Optional, Tuple, Union

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.flow_matching.conditional_probability_path import OTCondProbPath
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.flow_matching.configuration_uncertainty_sampler import CrossLikLaplaceSamplerConfig
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs.eval_uncertainty_estimation import EvalUncertaintyEstimationPipelineConfig
from lerobot.configs.visualize_laplace import VisualizeLaplacePipelineConfig


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
    
def collate_flow_matching_batch(
    batch: List[Tuple[Tuple[Tensor, Tensor, Dict[str, Tensor]], Tensor]]
) -> Tuple[FlowMatchingInput, Tensor]:
    """
    Collate a list of ((interp_traj, time, observation), target_vel) samples
    into a FlowMatchingInput batch and a target batch.

    Args:
        batch: List of ((interp_traj, time, observation), target_vel) pairs from
            the dataset.

    Returns:
        - FlowMatchingInput packed with interp_traj, time and observation.
        - Batch of target velocities.
    """
    # Separate inputs and targets
    raw_inputs, targets = zip(*batch)  
    interp_trajs, times, observations = zip(*raw_inputs)

    # Build a dict whose values are stacked across samples:
    # {
    #   "observation.state": (batch_size, ...),
    #   "observation.images": (batch_size, ...),
    #   ...
    # }
    obs_keys = observations[0].keys()
    observation_batch = {
        k: torch.stack([obs[k] for obs in observations])
        for k in obs_keys
    }

    # Stack into batched tensors
    batch_input = FlowMatchingInput(
        interp_traj = torch.stack(interp_trajs),
        time = torch.stack(times),
        observation = observation_batch,
    )
    target_batch = torch.stack(targets)
    target_batch = target_batch.flatten(start_dim=1)

    return batch_input, target_batch

class LaplaceFlowMatchingCalibrationDataset(Dataset):
    """
    Dataset for Laplace approximation calibration of a Flow Matching model including RGB encoder.
    """
    def __init__(
        self,
        interp_traj: torch.Tensor,
        times: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        target_vel: torch.Tensor,
    ):
        dataset_size = len(interp_traj)
        # Every tensor in the observation-dict must have the same size as the dataset
        assert all(v.size(0) == dataset_size for v in observations.values()), (
            f"Mismatch between dataset size ({dataset_size}) and at least one "
            f"observation tensor ({[v.size(0) for v in observations.values()]})"
        )
        # Times and target velocities must share the same dataset size
        assert len(times) == dataset_size, (
            f"Inconsistent lengths: interp_traj={dataset_size} vs times={len(times)}"
        )
        assert len(target_vel) == dataset_size, (
            f"Inconsistent lengths: interp_traj={dataset_size} vs target_vel={len(target_vel)}"
        )
        self.interp_traj = interp_traj
        self.times = times
        self.observations = observations
        self.target_vel = target_vel

    def __len__(self) -> int:
        return len(self.interp_traj)
    
    def __getitem__(self, idx):
        """
        Retrieve a single calibration sample which matches the 'laplace.LaPlace.fit()' method.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Velocity field model inputs including:
                - Interpolated trajectory based on some random noise sample and a target action
                using an optimal transport conditional probability path.
                - Time step along the flow path.
                - Input observations including the agent's state, RGB images of the environment
                  and/or the environment state.
            Target velocity for the input sample.
        """
        observation_i = {k: v[idx] for k, v in self.observations.items()}
        return (
            self.interp_traj[idx],
            self.times[idx],
            observation_i,
        ), self.target_vel[idx]

@torch.no_grad()
def create_laplace_flow_matching_calib_loader(
    cfg: Union[EvalUncertaintyEstimationPipelineConfig, VisualizeLaplacePipelineConfig],
    policy: PreTrainedPolicy,
) -> DataLoader:
    """
    Build a data loader for Laplace approximation calibration of a flow matching model
    including RGB encoder.

    Args:
        cfg: Configuration object providing dataset construction parameters.
        policy: A trained flow matching policy.

    Returns:
        A data loader over a small calibration set yielding batches of the form
        ((interpolated_trajectory, time_step, observation), target_velocity).
        This corresponds to the input and target output of the flow matching model
        including RGB encoder. This data loader matches the 'laplace.LaPlace.fit()' method.
    """
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device)

    batch_size = cfg.uncertainty_sampler.cross_likelihood_laplace_sampler.batch_size
    num_workers = cfg.uncertainty_sampler.cross_likelihood_laplace_sampler.num_workers
    train_dataset = make_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    disk_dir = Path(cfg.output_dir) / "laplace_calib_tmp"
    disk_dir.mkdir(parents=True, exist_ok=True)  

    calib_fraction = cfg.uncertainty_sampler.cross_likelihood_laplace_sampler.calib_fraction
    num_calib_samples = int(calib_fraction * len(train_dataset))
    samples_collected = 0
    pbar = trange(
        num_calib_samples,
        desc="Building Laplace calibration dataset",
        unit="sample",
        leave=False,
    )

    policy.eval()
    for batch in train_loader:
        if samples_collected >= num_calib_samples:
            break
        
        # Move all tensors in the batch to the correct device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)

        # Normalize inputs and targets
        batch = policy.normalize_inputs(batch)
        if cfg.policy.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in cfg.policy.image_features], dim=-4
            )
            if cfg.policy.n_obs_steps == 1:
                batch["observation.images"] = batch["observation.images"].unsqueeze(1)
        batch = policy.normalize_targets(batch)

        # Get ground-truth trajectory (x_1)
        trajectory = batch["action"]
        
        # Sample a random time for each item in the batch.
        times = torch.rand(
            size=(trajectory.shape[0],),
            device=device,
        )

        # Sample from conditional probability path and vector field.
        ot_cond_prob_path = OTCondProbPath()
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

        # Create storage tensors on first batch
        if not storage_created:
            interp_traj_tensor = torch.empty(
                (num_calib_samples, *interpolated_trajectory.shape[1:]),
                dtype=interpolated_trajectory.dtype
            )
            time_tensor = torch.empty(num_calib_samples, dtype=times.dtype)
            target_vel_tensor = torch.empty(
                (num_calib_samples, *target_vel.shape[1:]),
                dtype=target_vel.dtype
            )
            # observation dict – allocate per key
            observation_tensor_dict = {
                k: torch.empty(
                    (num_calib_samples, *v.shape[1:]),
                    dtype=v.dtype
                )
                for k, v in observation.items()
            }
            storage_created = True

        end_index = write_index + batch_size
        interp_traj_tensor[write_index:end_index] = interpolated_trajectory.cpu()
        time_tensor[write_index:end_index] = times.cpu()
        target_vel_tensor[write_index:end_index] = target_vel.cpu()
        for k, v in observation.items():
            observation_tensor_dict[k][write_index:end_index] = v

        write_index = end_index
        samples_collected = write_index
        pbar.update(batch_size)

    # Trim in case we over-collected by a few samples
    interp_traj_tensor = interp_traj_tensor[:num_calib_samples]
    time_tensor = time_tensor[:num_calib_samples]
    target_vel_tensor = target_vel_tensor[:num_calib_samples]
    observation_tensor_dict = {
        k: v[:num_calib_samples] for k, v in observation_tensor_dict.items()
    }

    calib_dataset = LaplaceFlowMatchingCalibrationDataset(
        interp_traj_tensor,
        time_tensor,
        observation_tensor_dict,
        target_vel_tensor,
    )
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type != "cpu",
        collate_fn=collate_flow_matching_batch
    )
    return calib_loader

def draw_laplace_flow_matching_model(
    laplace_posterior: BaseLaplace, 
    flow_matching_model: nn.Module,
    target_modules: list[str],
    generator: Optional[torch.Generator] = None,
) -> nn.Module:
    """
    Sample a single set of weights from a fitted Laplace posterior and
    insert it into a copy of the MAP flow matching model.

    Args:
        laplace_posterior: A fitted Laplace posterior.
        flow_matching_model: The original MAP flow matching model. It is not modified,
            but its architecture is only used as a template for the copy that will receive
            the sampled weights. 
        target_moodules: Modules the Laplace approximation to applied to.
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
}

def resolve_repo_dir(repo_id: str) -> str:
    try:
        return REPO_DIR_MAP[repo_id]
    except KeyError:
        raise ValueError(
            f"Unknown repo_id {repo_id!r}. Expected one of {list(REPO_DIR_MAP)!r}"
        )

def make_laplace_path(
    repo_id: str,
    scope: str,
    calib_fraction: float,
    batch_size: int,
) -> Path:
    """
    Build (and create) the on-disk path where we save/load a Laplace posterior.
    """
    base_dir = Path("outputs") / "laplace_posterior"
    subdir = resolve_repo_dir(repo_id)
    out_dir = base_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    pct = int(calib_fraction * 100)
    fname = f"laplace_{scope}_frac{pct}pct_bs{batch_size}.bin"
    return out_dir / fname

def get_laplace_posterior(
    cfg: CrossLikLaplaceSamplerConfig,
    flow_matching_model: nn.Module,
    laplace_calib_loader: Optional[torch.utils.data.DataLoader],
    laplace_path: str,
) -> Tuple[Laplace, List[str]]:
    """
    Build (or load) a diagonal Laplace posterior for a sub-network of a
    flow matching model.
    
    Args:
        cfg: Sampler-specific settings.
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
    if cfg.laplace_scope in ["velocity_last", "both"]:
        flow_matching_model.unet.final_conv[1] = PointwiseConv1dToLinear(
            flow_matching_model.unet.final_conv[1]
        )
        laplace_approx_targets.append("unet.final_conv.1.linear_layer")

    if cfg.laplace_scope in ["rgb_last", "both"]:
        laplace_approx_targets.append("rgb_encoder.out")

    if not laplace_approx_targets:
        raise ValueError(
            f"Unknown laplace_scope={cfg.laplace_scope}. "
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
        logging.info(f"Fitting new Laplace posterior.")
        if laplace_calib_loader is None:
            raise ValueError("Calibration loader is required to fit Laplace.")
        laplace_posterior.fit(laplace_calib_loader)

        logging.info(f"Save Laplace posterior to {laplace_path}")
        torch.save(laplace_posterior.state_dict(), laplace_path)

    return laplace_posterior, laplace_approx_targets