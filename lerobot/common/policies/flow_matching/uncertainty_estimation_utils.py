import copy
import torch

from dataclasses import dataclass
from laplace.baselaplace import BaseLaplace
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils import vector_to_parameters
from tqdm.auto import trange
from typing import Dict, List, Optional, Tuple

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.flow_matching.conditional_probability_path import OTCondProbPath
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs.eval_uncertainty_estimation import EvalUncertaintyEstimationPipelineConfig


@dataclass(frozen=True)
class VectorFieldInput:
    """
    Container for one batch of inputs to a flow-matching velocity network.
    """
    # Interpolated trajectory based on some random noise sample and a target action
    # using an optimal transport conditional probability path.
    interp_traj: torch.Tensor # Shape: (batch_size, horizon, action_dim)
    # Time step along the flow path.
    time: torch.Tensor # Shape: (batch_size,)
    # Conditioning vector based encoding of input observations.
    global_cond: torch.Tensor # Shape: (batch_size, cond_dim)

    def to(self, *args, **kwargs) -> "VectorFieldInput":
        """
        Return a copy of this VectorFieldInput with all contained tensors moved
        or cast.
        """
        return VectorFieldInput(
            interp_traj = self.interp_traj.to(*args, **kwargs),
            time = self.time.to(*args, **kwargs),
            global_cond = self.global_cond.to(*args, **kwargs),
        )

    def detach(self) -> "VectorFieldInput":
        return VectorFieldInput(
            self.interp_traj.detach(),
            self.time.detach(),
            self.global_cond.detach()
        )

    def cpu(self) -> "VectorFieldInput":
        return self.to("cpu")


class VelocityModelWrapper(nn.Module):
    """
    Wraps a Flow Matching velocity model so that laplace-torch can call it
    with a single VectorFieldInput object.
    """
    def __init__(self, velocity_model: nn.Module) -> None:
        super().__init__()
        self.base_model = velocity_model

    def forward(self, sample: VectorFieldInput):
        # Unpack VectorFieldInput attributes and forward them to the velocity model
        return self.base_model(sample.interp_traj, sample.time, sample.global_cond)
    

class LaplaceVelocityCalibrationDataset(TensorDataset):
    """
    Dataset for Laplace approximation calibration of a Flow Matching velocity field.
    """
    def __getitem__(self, index):
        """
        Retrieve a single calibration sample which matches the 'laplace.LaPlace.fit()' method.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Velocity field model inputs including:
                - Interpolated trajectory based on some random noise sample and a target action
                using an optimal transport conditional probability path.
                - Time step along the flow path.
                - Conditioning vector based encoding of input observations.
            Target velocity for the input sample.
        """
        interp_traj, time, global_cond, target_vel = super().__getitem__(index)
        
        return (interp_traj, time, global_cond), target_vel

def collate_vectorfield_batch(
    batch: List[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]]
) -> Tuple[VectorFieldInput, Tensor]:
    """
    Collate a list of ((interp_traj, time, global_cond), target_vel) samples
    into a VectorFieldInput batch and a target batch.

    Args:
        batch: List of ((interp_traj, time, global_cond), target_vel) pairs from
            the dataset.

    Returns:
        - VectorFieldInput packed with interp_traj, time and global_cond.
        - Batch of target velocities.
    """
    # Separate inputs and targets
    raw_inputs, targets = zip(*batch)  
    interp_trajs, times, global_conds = zip(*raw_inputs)

    # Stack into batched tensors
    batch_input = VectorFieldInput(
        interp_traj = torch.stack(interp_trajs),
        time = torch.stack(times),
        global_cond = torch.stack(global_conds),
    )
    batch_targets = torch.stack(targets)

    return batch_input, batch_targets

@torch.no_grad()
def create_laplace_velocity_calib_loader(
    cfg: EvalUncertaintyEstimationPipelineConfig,
    policy: PreTrainedPolicy,
    calib_fraction: float = 0.1,
) -> DataLoader:
    """
    Build a data loader for Laplace approximation calibration of a flow matching
    velocity model.

    Args:
        cfg: Configuration object providing dataset construction parameters.
        policy: A trained flow matching policy.
        calib_fraction: Fraction of the full training set to use for calibration.

    Returns:
        A data loader over a small calibration set yielding batches of the form
        ((interpolated_trajectory, time_step, global_condition), target_velocity).
        This corresponds to the input and target output of the flow matching velocity model.
        This data loader matches the 'laplace.LaPlace.fit()' method.
    """
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device)

    train_dataset = make_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # Accumulate velocity fieldd inputs and targets for Laplace calibration dataset
    interp_traj_list: list[torch.Tensor] = []
    time_list: list[torch.Tensor] = []
    cond_list: list[torch.Tensor] = []
    target_vel_list: list[torch.Tensor] = []

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

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = policy.flow_matching.prepare_global_conditioning(batch)  # (B, global_cond_dim)

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

        # Accumulate tensors
        interp_traj_list.append(interpolated_trajectory.cpu())
        time_list.append(times.cpu())
        cond_list.append(global_cond.cpu())
        target_vel_list.append(target_vel.cpu())

        samples_added = trajectory.shape[0]
        samples_collected += samples_added
        pbar.update(samples_added)

    # Stack velocity fieldd inputs and targets and wrap in data loader
    interp_traj_tensor = torch.cat(interp_traj_list)[:num_calib_samples]
    time_tensor = torch.cat(time_list)[:num_calib_samples]
    cond_tensor = torch.cat(cond_list)[:num_calib_samples]
    target_vel_tensor = torch.cat(target_vel_list)[:num_calib_samples]
    calib_dataset = LaplaceVelocityCalibrationDataset(
        interp_traj_tensor,
        time_tensor,
        cond_tensor,
        target_vel_tensor,
    )
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=device.type != "cpu",
        collate_fn=collate_vectorfield_batch
    )
    return calib_loader


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
    cfg: EvalUncertaintyEstimationPipelineConfig,
    policy: PreTrainedPolicy,
    calib_fraction: float,
) -> DataLoader:
    """
    Build a data loader for Laplace approximation calibration of a flow matching model
    including RGB encoder.

    Args:
        cfg: Configuration object providing dataset construction parameters.
        policy: A trained flow matching policy.
        calib_fraction: Fraction of the full training set to use for calibration.

    Returns:
        A data loader over a small calibration set yielding batches of the form
        ((interpolated_trajectory, time_step, observation), target_velocity).
        This corresponds to the input and target output of the flow matching model
        including RGB encoder. This data loader matches the 'laplace.LaPlace.fit()' method.
    """
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device)

    train_dataset = make_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # Accumulate velocity fieldd inputs and targets for Laplace calibration dataset
    interp_traj_list: list[torch.Tensor] = []
    time_list: list[torch.Tensor] = []
    obs_list: list[torch.Tensor] = []
    target_vel_list: list[torch.Tensor] = []

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

        # Accumulate tensors
        interp_traj_list.append(interpolated_trajectory.cpu())
        time_list.append(times.cpu())
        if len(obs_list) == 0:
            obs_list = {k: [v] for k, v in observation.items()}
        else:
            for k, v in observation.items():
                obs_list[k].append(v)
        target_vel_list.append(target_vel.cpu())
        samples_added = trajectory.shape[0]
        samples_collected += samples_added
        pbar.update(samples_added)

    # Stack velocity fieldd inputs and targets and wrap in data loader
    interp_traj_tensor = torch.cat(interp_traj_list)[:num_calib_samples]
    time_tensor = torch.cat(time_list)[:num_calib_samples]
    observation_tensor_dict = {
        k: torch.cat(v_list)[:num_calib_samples]
        for k, v_list in obs_list.items()
    }
    target_vel_tensor = torch.cat(target_vel_list)[:num_calib_samples]
    calib_dataset = LaplaceFlowMatchingCalibrationDataset(
        interp_traj_tensor,
        time_tensor,
        observation_tensor_dict,
        target_vel_tensor,
    )
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
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
    for name, m in flow_matching_model.named_modules():
        has_grad = any(p.requires_grad for p in m.parameters(recurse=False))

        if has_grad and name not in target_modules:
            raise RuntimeError(
                f"Module {name} needs deactivated gradients to properly assign "
                "weights sampled from Laplace posterior."
            )
        if (not has_grad) and name in target_modules:
            raise RuntimeError(
                f"Module {name} needs activated gradients to properly assign "
                "weights sampled from Laplace posterior."
            )

    # Draw weights from the Laplace posterior
    laplace_model_weights = laplace_posterior.sample(
        n_samples=1,
        generator=generator
    ).squeeze(0)

    # Copy the MAP model so we never mutate the original
    laplace_flow_matching_model = copy.deepcopy(flow_matching_model)

    # Collect the parameters that were *in* the posterior
    target_params = [p for p in laplace_flow_matching_model.parameters() if p.requires_grad]

    # Consistency check – avoids silent weight mis-alignment.
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