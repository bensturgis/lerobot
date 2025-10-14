from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.common.flow_matching.ode_solver import ODESolver

from .configuration_visualizer import VectorFieldVisConfig
from .utils import add_actions
from .visualizer import FlowMatchingVisualizer


class VectorFieldVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting vector fields.
    """
    def __init__(
        self,
        config: VectorFieldVisConfig,
        model: BaseFlowMatchingAdapter,
        output_root: Optional[Union[Path, str]],
        save: bool = True,
        create_gif: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            config: Visualizer-specific settings.
        """
        super().__init__(
            model=model,
            save=save,
            output_root=output_root,
            create_gif=create_gif,
            verbose=verbose,
        )
        if len(config.action_dims) not in (2, 3):
            raise ValueError(
                "The vector-field visualisation supports 2D and 3D only, "
                f"(got action_dims = {config.action_dims}."
            )

        self.action_dims = config.action_dims
        self.action_dim_names = config.action_dim_names
        self.action_steps = config.action_steps or []
        self.min_action = config.min_action
        self.max_action = config.max_action
        self.grid_size = config.grid_size
        # Default time_grid is list [0.05, 0.1, ..., 1.0]
        self.time_grid = list(np.linspace(0, 1, 21)) if config.time_grid is None else config.time_grid
        self.show = config.show
        self.gif_name_base = "vector_field_animation"
        self.vis_type = "vector_field"

    def visualize(
        self, observation: Dict[str, Tensor], generator: Optional[torch.Generator] = None, **kwargs
    ):
        """
        Visualize the 2D action vector field produced by a flow matching policy at a given time.

        Args:
            observation: Info about the environment used to create the conditioning for
                the flow matching model.
            generator: PyTorch random number generator.
        """
        device = self.model.device
        dtype = self.model.dtype

        self.run_dir = self._update_run_dir()

        # Initialize ODE solver
        ode_solver = ODESolver()

        visualize_actions: bool = kwargs.get("visualize_actions", True)
        action_data = kwargs.get("actions", {})
        if visualize_actions and not action_data:
            # If no actions for visualization were passed in, sample some
            num_samples = 50 if len(self.action_dims) == 2 else 100
        else:
            # Only sample a single action sequence to create the vector field
            num_samples = 1

        # Build the velocity function conditioned on the current observation
        conditioning = self.model.prepare_conditioning(observation, num_samples)
        velocity_fn = self.model.make_velocity_fn(conditioning=conditioning)

        # Sample noise vectors from prior
        noise_sample = self.model.sample_prior(
            num_samples=num_samples,
            generator=generator
        )

        # Sample action sequences
        action_samples = ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=velocity_fn,
            method=self.model.ode_solver_config["solver_method"],
            step_size=self.model.ode_solver_config["step_size"],
            atol=self.model.ode_solver_config["atol"],
            rtol=self.model.ode_solver_config["rtol"],
        )

        action_data_colors: List[str] = []
        if visualize_actions and not action_data:
            action_data["action_samples"] = action_samples[1:]
            action_data_colors = ["red"]

        if "Base Action" not in action_data:
            action_data["Base Action"] = action_samples[0].unsqueeze(0)
            action_data_colors.append("cyan")

        # Build a 1-D lin-space once and reuse it for every axis we need
        axis_lin = np.linspace(self.min_action, self.max_action, self.grid_size)

        # Create the grids
        if len(self.action_dims) == 2:
            x_grid, y_grid = np.meshgrid(axis_lin, axis_lin, indexing="xy")
            x_dim, y_dim = self.action_dims
        else:
            x_grid, y_grid, z_grid = np.meshgrid(axis_lin, axis_lin, axis_lin, indexing="xy")
            x_dim, y_dim, z_dim = self.action_dims

        if len(self.action_steps) == 0:
            self.action_steps = list(range(self.model.horizon))
        action_steps_idx = torch.randint(
            low=0,
            high=len(self.action_steps),
            size=(1,),
            generator=generator,
            device=device,
        ).item()
        action_step = self.action_steps[action_steps_idx]
        positions = action_data["Base Action"].repeat(x_grid.size, 1, 1)
        positions[:, action_step, x_dim] = torch.tensor(x_grid.ravel(), dtype=dtype, device=device)
        positions[:, action_step, y_dim] = torch.tensor(y_grid.ravel(), dtype=dtype, device=device)
        if len(self.action_dims) == 3:
            positions[:, action_step, z_dim] = torch.tensor(z_grid.ravel(), dtype=dtype, device=device)

        # Build a condition vector tensor whose batch size is the number of grid points
        num_grid_points = positions.shape[0]

        # Rebuild the velocity function conditioned on the current observation
        conditioning = self.model.prepare_conditioning(observation, num_grid_points)
        velocity_fn = self.model.make_velocity_fn(conditioning=conditioning)

        # Compute the max velocity norm over the timesteps for coloring
        max_velocity_norm = float('-inf')
        for time in reversed(self.time_grid):
            time = torch.tensor(time, device=device, dtype=dtype)
            with torch.no_grad():
                velocities = velocity_fn(x_t=positions, t=time)
            norms = torch.norm(velocities[:, action_step, self.action_dims], dim=1)
            cur_max_velocity_norm = norms.max().item()
            if cur_max_velocity_norm <= max_velocity_norm:
                break
            max_velocity_norm = cur_max_velocity_norm

        for time in self.time_grid:
            # Compute velocity at grid points and current time as given by flow matching velocity model
            time = torch.tensor(time, device=device, dtype=dtype)
            with torch.no_grad():
                velocities = velocity_fn(x_t=positions, t=time)

            if len(self.action_dims) == 2:
                fig = self._create_vector_field_plot_2d(
                    x_positions=x_grid.reshape(-1),
                    y_positions=y_grid.reshape(-1),
                    x_velocities=velocities[:, action_step, x_dim].detach().cpu().numpy(),
                    y_velocities=velocities[:, action_step, y_dim].detach().cpu().numpy(),
                    limits=(self.min_action, self.max_action),
                    action_step=action_step,
                    time=time,
                    max_velocity_norm=max_velocity_norm,
                    uncertainty=kwargs.get("uncertainty")
                )
            else:
                fig = self._create_vector_field_plot_3d(
                    x_positions=x_grid.reshape(-1),
                    y_positions=y_grid.reshape(-1),
                    z_positions=z_grid.reshape(-1),
                    x_velocities=velocities[:, action_step, x_dim].cpu().numpy(),
                    y_velocities=velocities[:, action_step, y_dim].cpu().numpy(),
                    z_velocities=velocities[:, action_step, z_dim].cpu().numpy(),
                    limits=(self.min_action, self.max_action),
                    action_step=action_step,
                    time=time,
                    max_velocity_norm=max_velocity_norm,
                    uncertainty=kwargs.get("uncertainty")
                )

            if visualize_actions:
                add_actions(
                    ax=fig.axes[0],
                    action_data=action_data,
                    action_step=action_step,
                    action_dims=self.action_dims,
                    colors=action_data_colors,
                )

            # Legend
            fig.axes[0].legend()

            if self.show:
                plt.show(block=True)
            if self.save:
                self._save_figure(fig, time=time)

            plt.close(fig)

        if self.create_gif:
            self._create_gif()

    def _create_vector_field_plot_3d(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        z_positions: np.ndarray,
        x_velocities: np.ndarray,
        y_velocities: np.ndarray,
        z_velocities: np.ndarray,
        limits: Tuple[float, float],
        action_step: int,
        time: float,
        max_velocity_norm: float,
        uncertainty: Optional[float] = None,
    ) -> plt.Figure:
        """
        Draw a 3D quiver plot for the vector field of a three action dimensions in a single
        action step of a whole action sequence at a given time.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        # Color arrows by velocity norm
        norms = np.sqrt(
            x_velocities**2 + y_velocities**2 + z_velocities**2
        )

        # Create quiver plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
        fig.canvas.manager.set_window_title("Visualization of Vector Field")
        quiv = ax.quiver(
            x_positions, y_positions, z_positions,
            x_velocities, y_velocities, z_velocities,
            array=norms,
            cmap='viridis',
            norm=plt.Normalize(vmin=0.0, vmax=max_velocity_norm),
            length=0.025,
            normalize=False,
            linewidth=0.7,
            arrow_length_ratio=0.2,
        )

        # Set axis limits
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.set_aspect('equal')

        # Colorbar and title
        cbar = fig.colorbar(quiv, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel('Velocity Norm', fontsize=12)

        # Title
        ax.set_title(f"Vector Field of Action Step {action_step+1} at t={time:.2f}", fontsize=16)

        # Axis labels
        if self.action_dim_names:
            x_label = self.action_dim_names[self.action_dims[0]]
            y_label = self.action_dim_names[self.action_dims[1]]
            z_label = self.action_dim_names[self.action_dims[2]]
        else:
            x_label = f"Action dimension {self.action_dims[0]}"
            y_label = f"Action dimension {self.action_dims[1]}"
            z_label = f"Action dimension {self.action_dims[2]}"
        ax.set_xlabel(x_label, fontsize=14, labelpad=8)
        ax.set_ylabel(y_label, fontsize=14, labelpad=8)
        ax.set_zlabel(z_label, fontsize=14, labelpad=8)

        if uncertainty:
            ax.text2D(
                0.02, 0.98,
                f"Uncertainty: {uncertainty:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8
                },
            )

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        plt.tight_layout()

        if was_interactive:
            plt.ion()

        return fig

    def _create_vector_field_plot_2d(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        x_velocities: np.ndarray,
        y_velocities: np.ndarray,
        limits: Tuple[float, float],
        action_step: int,
        time: float,
        max_velocity_norm: float,
        uncertainty: Optional[float] = None,
    ) -> plt.Figure:
        """
        Draw a 2D quiver plot for the vector field of a two action dimensions in a single
        action step of a whole action sequence at a given time.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        # Color arrows by velocity norm
        norms = np.sqrt(
            x_velocities**2 + y_velocities**2
        )

        # Create quiver plot
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.canvas.manager.set_window_title("Visualization of Vector Field")
        # Presentation: scale=9, width=0.005
        quiv = ax.quiver(
            x_positions, y_positions,
            x_velocities, y_velocities,
            norms, cmap='viridis',
            norm=plt.Normalize(vmin=0.0, vmax=max_velocity_norm),
            angles='xy', scale=50,
            scale_units='xy', width=0.005,
        )

        # Set axis limits
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')

        # Colorbar and title
        cbar = fig.colorbar(quiv, ax=ax, shrink=0.95)
        # Presentation: fontsize=32, labelpad=12
        cbar.ax.set_ylabel('Velocity Norm', fontsize=12)
        # Presentation
        # cbar.ax.tick_params(labelsize=28)
        # cbar.set_ticks(np.arange(0.0, 2.01, 0.5))

        # Title
        ax.set_title(
            f"Vector Field of Action Step {action_step+1} at t={time:.2f}",
            fontsize=16,
            pad=12
        )

        # Axis labels
        if self.action_dim_names:
            x_label = self.action_dim_names[self.action_dims[0]]
            y_label = self.action_dim_names[self.action_dims[1]]
        else:
            x_label = f"Action dimension {self.action_dims[0]}"
            y_label = f"Action dimension {self.action_dims[1]}"
        # Presentation: fontsize=34, labelpad=4
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

        if uncertainty:
            ax.text(
                0.02, 0.98,
                f"Uncertainty: {uncertainty:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8
                },
            )

        ax.tick_params(axis='both', labelsize=12)
        # Presentation
        # ax.tick_params(
        #     axis='both',
        #     which='both',
        #     labelbottom=False,
        #     labelleft=False
        # )
        ax.grid(True)
        plt.tight_layout()

        if was_interactive:
            plt.ion()

        return fig

    def get_figure_filename(self, **kwargs) -> str:
        """Get the figure filename for the current evaluation time."""
        if "time" not in kwargs:
            raise ValueError(
                "`time` must be provided to get filename of vector field figure."
            )
        time = kwargs["time"]

        return f"vector_field_{int(time * 100):03d}.png"
