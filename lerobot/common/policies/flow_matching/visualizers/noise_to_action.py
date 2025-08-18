from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingConditionalUnet1d
from lerobot.common.policies.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
)
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.configs.default import NoiseToActionVisConfig

from .base import FlowMatchingVisualizer
from .utils import add_actions, compute_axis_limits


class NoiseToActionVisualizer(FlowMatchingVisualizer):
    """Visualizer that the transformation of an initial noise into an action sequence."""
    def __init__(
        self,
        cfg: NoiseToActionVisConfig,
        flow_matching_cfg: FlowMatchingConfig,
        velocity_model: FlowMatchingConditionalUnet1d,
        output_root: Optional[Union[Path, str]],
        save: bool = True,
        create_gif: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            cfg: Visualizer-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            save=save,
            output_root=output_root,
            create_gif=create_gif,
            verbose=verbose,
        )
        self.device = get_device_from_parameters(self.velocity_model)
        self.dtype = get_dtype_from_parameters(self.velocity_model)
        if len(cfg.action_dims) != 2:
            raise ValueError(
                "The noise-to-action visualization supports 2D only, "
                f"(got action_dims = {cfg.action_dims}."
            )
        self.horizon = flow_matching_cfg.horizon
        self.action_dims = cfg.action_dims
        self.axis_limits = cfg.axis_limits
        self.action_dim_names = cfg.action_dim_names
        self.num_samples = cfg.num_samples
        self.show = cfg.show
        self.gif_name_base = "noise_to_action_animation"
        self.vis_type = "noise_to_action"

        # Initialize ODE solver
        self.ode_solver = ODESolver(self.velocity_model)

        # Create time grid for solving the ODE including the times where the noisy actions get visualized
        self.ode_eval_times = cfg.ode_eval_times.to(self.device)
        ode_solver_method = self.flow_matching_cfg.ode_solver_method
        if ode_solver_method in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = self.ode_solver.make_sampling_time_grid(
                step_size=self.flow_matching_cfg.ode_step_size,
                extra_times=self.ode_eval_times,
                device=self.device,
            )
        elif ode_solver_method in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = self.ode_eval_times
            if self.sampling_time_grid[-1].item() != 1.0:
                self.sampling_time_grid = torch.cat([self.sampling_time_grid, self.sampling_time_grid.new_tensor([1.0])])
        else:
            raise ValueError(f"Unknown ODE method: {ode_solver_method}")

    def plot_noise_to_action_overlays(
        self,
        action_overlays: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Create one figure per evaluation time and plot multiple overlays together.

        Each overlay dict should contain 'label', 'ode_states' with shape (num_samples, timesteps,
        horizon, action_dim), 'step_labels' (optional), 'colors' (optional), 'zorder' (optional),
        'scale' (optional), 'marker' (optional).
        """
        if not action_overlays:
            return

        # Compute axis limits once to create plots of equal size
        if self.axis_limits is None:
            combined_ode_states = torch.cat([ov["ode_states"] for ov in action_overlays], dim=0)
            self.axis_limits = compute_axis_limits(
                ode_states=combined_ode_states, action_dims=self.action_dims
            )

        # Text labels for axes
        if self.action_dim_names:
            labels = [self.action_dim_names[d] for d in self.action_dims]
        else:
            labels = [f"Action dimension {d}" for d in self.action_dims]

        # Create a separate figure for each time step
        for time_idx, time in enumerate(self.ode_eval_times):
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot all overlays
            for action_step in range(self.horizon):
                for spec in action_overlays:
                    # Pick the slice at this time index
                    actions_t = spec["ode_states"][:, time_idx, :, :]  # (num_samples, horizon, action_dim)
                    step_label = spec["step_labels"][action_step] if "step_labels" in spec else None
                    step_color = spec["colors"][action_step] if "colors" in spec else None
                    add_actions(
                        ax=ax,
                        action_data={spec["label"]: actions_t},
                        action_step=action_step,
                        action_dims=self.action_dims,
                        colors=[step_color],
                        step_label=step_label,
                        zorder=spec.get("zorder", 3),
                        scale=spec.get("scale", 40.0),
                        marker=spec.get("marker", "o")
                    )

            # Axis labels
            ax.set_xlabel(labels[0], fontsize=14)
            ax.set_ylabel(labels[1], fontsize=14)
            # Axis limits
            ax.set_xlim(*self.axis_limits[0])
            ax.set_ylim(*self.axis_limits[1])
            ax.set_aspect("equal")
            # Title
            ax.set_title(f"Noisy Actions at t={float(time):.2f}", fontsize=16)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(True)
            plt.tight_layout()

            # Legend
            ax.legend()

            if self.show:
                plt.show(block=True)

            if self.save:
                self._save_figure(fig, time=time)

            plt.close(fig)

    def visualize(
        self, global_cond: Tensor, generator: Optional[torch.Generator] = None, **kwargs
    ):
        """
        Visualize noise to action transformation for the specified action dimensions.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        self.run_dir = self._update_run_dir()
        
        # Sample noise from prior
        action_dim = self.flow_matching_cfg.action_feature.shape[0]
        noise_sample = torch.randn(
            size=(self.num_samples, self.flow_matching_cfg.horizon, action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator
        )
        
        # Sample paths from the ODE
        ode_states = self.ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(self.num_samples, 1),
            step_size=self.flow_matching_cfg.ode_step_size,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        ) # Shape: (timesteps, num_paths, horizon, action_dim)

        # Extract the noisy action to visualize
        ode_eval_states, self.ode_eval_times = self.ode_solver.select_ode_states(
            time_grid=self.sampling_time_grid,
            ode_states=ode_states,
            requested_times=self.ode_eval_times,
        )
        ode_eval_states = ode_eval_states.transpose(0, 1) # Shape: (num_samples, timesteps, horizon, action_dim)
        
        horizon = ode_eval_states.shape[2]
        # step_labels = ("t", *[f"t+{k}" for k in range(1, horizon)])
        colors = cm.get_cmap('plasma')(torch.arange(horizon) / (horizon - 1))
        noisy_action_overlay = [
            {
                "label": "Noisy Actions",
                "ode_states": ode_eval_states,
                "colors": colors,
            },
        ]
        self.plot_noise_to_action_overlays(action_overlays=noisy_action_overlay)

        if self.create_gif:
            self._create_gif()

        if was_interactive:
            plt.ion()

    def get_figure_filename(self, **kwargs) -> str:
        """Get the figure filename for the current evaluation time."""
        if "time" not in kwargs:
            raise ValueError(
                "`time` must be provided to get filename of noise-to-action figure."
            )
        time = kwargs["time"]

        return f"noise_to_action_{int(time * 100):03d}.png"