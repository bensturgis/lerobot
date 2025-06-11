import torch

from dataclasses import dataclass
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange
from typing import List, Tuple

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.flow_matching.conditional_probability_path import OTCondProbPath
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
        self.velocity_model = velocity_model

    def forward(self, sample: VectorFieldInput):
        # Unpack VectorFieldInput attributes and forward them to the velocity model
        return self.velocity_model(sample.interp_traj, sample.time, sample.global_cond)
    

class LaplaceCalibrationDataset(TensorDataset):
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
def create_laplace_calibration_dataloader(
    cfg: EvalUncertaintyEstimationPipelineConfig,
    policy,
    calib_fraction: float = 0.1,
) -> DataLoader:
    """
    Build a data loader for Laplace approximation calibration of a flow matching model.

    Args:
        cfg: Configuration object providing dataset construction parameters.
        policy: A trained flow matching policy which exposes.
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
    calib_dataset = LaplaceCalibrationDataset(
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