from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import default_collate

from lerobot.constants import ACTION
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.flow_matching.modelling_flow_matching import FlowMatchingModel
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.utils import get_safe_torch_device

from .laplace_wrapper import LaplaceBatch, LaplaceWrapper


@dataclass
class FlowMatchingLaplaceBatch(LaplaceBatch):
    """
    Container for one batch of inputs to a flow matching model including RGB encoder.
    """
    # Interpolated trajectory based on some random noise sample and a target action
    # using an optimal transport conditional probability path.
    interp_traj: torch.Tensor # Shape: (batch_size, horizon, action_dim)
    # Time step along the flow path.
    time: torch.Tensor # Shape: (batch_size,)
    # Input observations of the environment.
    observation: Dict[str, torch.Tensor]

    def to(self, *args, **kwargs) -> "FlowMatchingLaplaceBatch":
        """
        Return a copy of this FlowMatchingInput with all contained tensors moved or cast.
        """
        return FlowMatchingLaplaceBatch(
            interp_traj = self.interp_traj.to(*args, **kwargs),
            time = self.time.to(*args, **kwargs),
            observation = {k: v.to(*args, **kwargs) for k, v in self.observation.items()}
        )

    def detach(self) -> "FlowMatchingLaplaceBatch":
        return FlowMatchingLaplaceBatch(
            interp_traj = self.interp_traj.detach(),
            time = self.time.detach(),
            observation = {k: v.detach() for k, v in self.observation.items()}
        )

    def cpu(self) -> "FlowMatchingLaplaceBatch":
        return self.to("cpu")

class FlowMatchingLaplaceWrapper(LaplaceWrapper):
    def __init__(self, config: FlowMatchingConfig, model: FlowMatchingModel):
        super().__init__(model=model, config=config)
        self.cond_prob_path = self.model.cond_prob_path

    def build_collate_fn(
        self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    ) -> Callable[[Dict[str, Tensor]], Tuple[FlowMatchingLaplaceBatch, Tensor]]:
        """
        Factory that builds a dataloader collate function tailored for laplace-torch
        calibration of a flow matching model.

        The returned method converts a list of raw samples coming from the training dataset into the
        pair (FlowMatchingLaplaceBatch, target_velocity) to fit a Laplace posterior using laplace-torch.

        Args:
            preprocessor: Preprocessor to apply to raw dataset samples.
        """
        # Check device is available
        device = get_safe_torch_device(self.device)

        def collate_fn(batch: Dict[str, Tensor]) -> Tuple[FlowMatchingLaplaceBatch, Tensor]:
            batch = default_collate(batch)
            batch = preprocessor(batch)

            if self.config.image_features:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch["observation.images"] = torch.stack(
                    [batch[key] for key in self.config.image_features], dim=-4
                )

                # Get ground-truth trajectory (x_1)
                trajectory = batch[ACTION]
                batch_size = trajectory.shape[0]

                # Sample a random time for each item in the batch
                times = torch.rand(
                    size=(batch_size,),
                    device=device,
                )

                # Sample from conditional probability path and vector field.
                interpolated_trajectory = self.cond_prob_path.sample(
                    x_1=trajectory,
                    t=times[:, None, None],
                )
                target_vel = self.cond_prob_path.velocity(
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
            input_batch = FlowMatchingLaplaceBatch(
                interp_traj=interpolated_trajectory,
                time=times,
                observation=observation,
            )
            target_batch = target_vel.flatten(start_dim=1)

            return input_batch, target_batch

        return collate_fn

    def forward(self, batch: FlowMatchingLaplaceBatch) -> Tensor:
        # Unpack FlowMatchingInput attributes and forward them to the velocity model
        output = self.model(batch.interp_traj, batch.time, batch.observation)
        return output.flatten(start_dim=1)

    def apply_laplace_scope(self, scope: str):
        """
        Configure the model for Laplace approximation under a given scope:
        - optionally convert final Conv1d to Linear (pointwise-equivalent),
        - freeze all params and unfreeze only the scoped submodules.

        Args:
            scope: Which part of the model to approximate
                - "velocity_last": final layer of the velocity model
                - "rgb_last": final layer of the RGB encoder
                - "both": apply Laplace on both layers jointly
        """
        # Select target modules for Laplace approximation
        laplace_approx_targets: list[str] = []
        if scope in ["velocity_last", "both"]:
            if isinstance(self.model.unet.final_conv[1], nn.Conv1d):
                self.model.unet.final_conv[1] = PointwiseConv1dToLinear(
                    self.model.unet.final_conv[1]
                )
            elif (
                not isinstance(self.model.unet.final_conv[1], nn.Linear) and
                not isinstance(self.model.unet.final_conv[1], PointwiseConv1dToLinear)
            ):
                raise ValueError(
                    "Expected final layer of velocity model to be nn.Conv1d or nn.Linear, "
                    f"got {type(self.model.unet.final_conv[1])}"
                )
            laplace_approx_targets.append("unet.final_conv.1.linear_layer")

        if scope in ["rgb_last", "both"]:
            laplace_approx_targets.append("rgb_encoder.out")

        if not laplace_approx_targets:
            raise ValueError(
                f"Unknown laplace_scope={scope}. "
                "Choose from ['velocity_last', 'rgb_last', 'both']"
            )
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Unfreeze only the selected subnetwork for Laplace fitting
        for name, module in self.model.named_modules():
            if name in laplace_approx_targets:
                for p in module.parameters():
                    p.requires_grad_(True)


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
