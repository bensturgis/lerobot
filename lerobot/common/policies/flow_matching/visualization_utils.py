import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from torch import nn, Tensor

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.ode_solver import ODESolver
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

class FlowMatchingVisualizer:
    """
    Visualizer for Flow Matching paths and vector fields of action sequences.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        action_dim_names: list | tuple | None = None,
        output_dir: Path | str | None = None
    ):
        """
        Args:
            config: Configuration object for Flow Matching settings.
            velocity_model: The learned flow matching velocity model.
            action_dim_names: Optional names for action dimensions (used for axis labels).
            output_dir: Optional output directory for saving figures.
        """
        self.config = config
        self.velocity_model = velocity_model
        self.action_dim_names = action_dim_names
        self.output_dir = output_dir

    # TODO:
    # - Add option to visualize 3D flows
    def visualize_flows(
        self,
        global_cond: Tensor,
        action_dims: list | tuple = (0, 1),
        action_steps: list | int | None = None,
        num_paths: int = 300,
        show: bool = False,
        save: bool = True,
        create_gif: bool = True,
    ):
        """
        Visualize flow fields for specified action steps and dimensions.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
            action_dims: Two action dimensions to plot.
            action_steps: Steps to visualize; defaults to all.
            num_paths: Number of trajectories per step.
            show: If True, display the plots.
            save: If True, save the plots to disk.
        """
        if global_cond.dim() == 1: # shape = (cond_dim,)
            global_cond = global_cond.unsqueeze(0)     # (1, cond_dim)
        elif global_cond.dim() == 2 and global_cond.size(0) == 1: # shape = (1, cond_dim)
            pass
        else:
            raise ValueError(
                f"Expected global_cond to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )
        
        if not isinstance(action_dims, (list, tuple)) or len(action_dims) != 2:
            raise ValueError(f"'action_dims' must be a list or tuple of exactly two elements, but got {action_dims}")

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)
        
        # Initialize ODE solver
        ode_solver = ODESolver(self.velocity_model)
        
        # Sample noise from prior
        noise_sample = torch.randn(
            size=(num_paths, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
        )
        
        # Create time grid
        time_grid = torch.arange(0.0, 1.0, 0.1, device=device)

        # Sample paths from the ODE
        paths = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(num_paths, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
            time_grid=time_grid,
            return_intermediates=True,
        ).transpose(0, 1)  # Shape: (num_paths, timesteps, horizon, action_dim)

        # Compute vector field (velocity at each position)
        vector_field = torch.empty_like(paths)
        for p in range(num_paths):
            path = paths[p]
            with torch.no_grad():
                path_velocities = self.velocity_model(
                    path,
                    time_grid,
                    global_cond.repeat(len(time_grid), 1)
                )
            vector_field[p] = path_velocities
        
        # Select the specific action step and dimensions
        positions = paths[..., action_dims]
        velocity_vectors = vector_field[..., action_dims]
        
        # Compute global axis limits to create plots of equal size
        x_positions, y_positions = positions[:, :, action_steps, 0], positions[:, :, action_steps, 1]
        x_min, x_max = x_positions.min().cpu(), x_positions.max().cpu()
        y_min, y_max = y_positions.min().cpu(), y_positions.max().cpu()
        margin_x = 0.05 * (x_max - x_min)
        margin_y = 0.05 * (y_max - y_min)
        x_lim = (x_min - margin_x, x_max + margin_x)
        y_lim = (y_min - margin_y, y_max + margin_y)

        # Visualize all action steps by default
        if action_steps is None:
            action_steps = list(range(self.config.horizon))

        # Create a separate figure for each action steo
        for action_step in action_steps:
            positions_single_action = positions[:, :, action_step, :]
            velocity_vectors_single_action = velocity_vectors[:, :, action_step, :]

            fig = self._create_flow_plot(
                positions_single_action, velocity_vectors_single_action,
                time_grid, num_paths,
                x_lim, y_lim,
                action_step, action_dims
            )

            # Show plot if requested
            if show:
                fig.show(block=True)

            # Save plot if requested
            if save:
                self._save_figure(fig, "flows", action_step)
        
        if create_gif:
            self._create_flows_gif(
                images_dir=self.output_dir,
                action_steps=action_steps,
            )

    def visualize_vector_field(
        self,
        global_cond: Tensor,
        min_action: np.array,
        max_action: np.array,
        grid_size: int = 50,
        time: float = 0.5,
        show: bool = True,
        save: bool = True,
    ):
        """
        Visualize the 2D action vector field produced by a flow matching policy at a given time.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
            min_action: Lower bounds for x and y (shape (2,)).
            max_action: Upper bounds for x and y (shape (2,)).
            grid_size: Number of grid points per axis.
            time: Time step in [0,1] at which to evaluate the velocity field.
            show: If True, display the plot.
            save: If True, save the figure to disk.
        """
        if global_cond.dim() == 1: # shape = (cond_dim,)
            global_cond = global_cond.unsqueeze(0)     # (1, cond_dim)
        elif global_cond.dim() == 2 and global_cond.size(0) == 1: # shape = (1, cond_dim)
            pass
        else:
            raise ValueError(
                f"Expected global_cond to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )
    
        if self.config.action_feature.shape[0] != 2 or self.config.horizon != 1:
            raise ValueError(
                "The vector-field visualisation requires action_dim = 2 and horizon = 1 "
                f"(got action_dim = {self.config.action_feature.shape[0]}, "
                f"horizon = {self.config.horizon})."
            )
        
        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)
        
        # Clamp values to [-3, +3]
        min_lim = np.minimum(min_action, -3.0)
        max_lim = np.maximum(max_action,  3.0)

        # Unpack minimum x- and y-coordinates
        x_min, y_min = min_lim
        x_max, y_max = max_lim

        # Create a 2D meshgrid in the range of [x_min, x_max] x [y_min, y_max]
        x_lin = np.linspace(x_min, x_max, grid_size)
        y_lin = np.linspace(y_min, y_max, grid_size)
        x_grid, y_grid = np.meshgrid(x_lin, y_lin, indexing="xy")
        
        # Flatten numpy grids and turn them into a (num_grid_points,1,2) torch tensor to input into
        # the velocity model
        positions = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1)
        positions = torch.from_numpy(positions).unsqueeze(1).to(device=device, dtype=dtype)

        # Build a time and condition vector tensor whose batch size is the number of grid points
        num_grid_points = positions.shape[0]
        time_batch = torch.full((num_grid_points,), time, device=device, dtype=dtype)
        global_cond_batch = global_cond.repeat(num_grid_points, 1)

        # Compute velocity at grid points as given by flow matching velocity model
        with torch.no_grad():
            velocities = self.velocity_model(positions, time_batch, global_cond_batch)

        fig = self._create_vector_field_plot(
            x_positions=x_grid.reshape(-1),
            y_positions=y_grid.reshape(-1),
            x_velocities=velocities[:, 0, 0].cpu().numpy(),
            y_velocities=velocities[:, 0, 1].cpu().numpy(),
            x_lim=(x_min, x_max),
            y_lim=(y_min, y_max),
            time=time,
        )

        # Show plot if requested
        if show:
            plt.show(block=True)

        # Save plot if requested
        if save:
            self._save_figure(fig, "vector_field")

    def _create_flow_plot(
        self,
        positions: Tensor,
        velocity_vectors: Tensor,
        time_grid: Tensor,
        num_paths: int,
        x_lim: tuple,
        y_lim: tuple,
        action_step: int,
        action_dims: list | tuple = (0, 1),
    ) -> plt.Figure:
        """
        Draw a quiver plot for the flow of a single action step.
        """
        # Extract x- and y-coordinates and velocities
        x = positions[..., 0].flatten().cpu()
        y = positions[..., 1].flatten().cpu()
        u = velocity_vectors[..., 0].flatten().cpu()
        v = velocity_vectors[..., 1].flatten().cpu()

        # Create quiver plot
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.manager.set_window_title("Visualization of Flows")
        quiv = ax.quiver(
            x, y, u, v, time_grid.repeat(num_paths).cpu(),
            angles='xy', scale=len(time_grid),
            scale_units='xy', width=0.004, cmap='viridis'
        )

        # Set consistent axis limits so the plots of all action steps have same size
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect('equal')

        # Colorbar and title
        cbar = fig.colorbar(quiv, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel('Time', fontsize=12)
        ax.set_title(f"Flow of Action Step {action_step} (Horizon: {self.config.horizon})",
                     fontsize=16)

        # Axis labels
        if self.action_dim_names:
            x_label = self.action_dim_names[action_dims[0]]
            y_label = self.action_dim_names[action_dims[1]]
        else:
            x_label = f"Action dimension {action_dims[0]}"
            y_label = f"Action dimension {action_dims[1]}"
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        plt.tight_layout()
        return fig
    
    def _create_vector_field_plot(
        self,
        x_positions: np.array,
        y_positions: np.array,
        x_velocities: np.array,
        y_velocities: np.array,
        x_lim: tuple,
        y_lim: tuple,
        time: float,
    ) -> plt.Figure:
        """
        Draw a quiver plot for the vector field of a single two-dimensional action at a
        given time.
        """
        # Create quiver plot
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.manager.set_window_title("Visualization of Vector Field")
        ax.quiver(
            x_positions, y_positions,
            x_velocities, y_velocities,
            angles='xy', scale=40,
            scale_units='xy', width=0.004,
            cmap='viridis'
        )

        # Set axis limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        
        # Title
        ax.set_title(f"Vector Field at t={time:.2f}", fontsize=16)

        # Axis labels
        if self.action_dim_names:
            x_label = self.action_dim_names[0]
            y_label = self.action_dim_names[1]
        else:
            x_label = f"Action dimension {0}"
            y_label = f"Action dimension {1}"
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        plt.tight_layout()
        
        return fig

    def _save_figure(
        self,
        fig: plt.Figure,
        vis_type: str,
        action_step: int | None =  None
    ) -> None:
        """
        Save the given figure as 'flows_action_##.png', skipping if it already exists.
        """
        # Create a new run folder if needed
        if self.output_dir is None:
            vis_dir = Path("outputs/visualizations/")
            base_dir = vis_dir / vis_type
            base_dir.mkdir(parents=True, exist_ok=True)
            run_idx = 1
            while (base_dir / f"{run_idx:04d}").exists():
                run_idx += 1
            self.output_dir = base_dir / f"{run_idx:04d}"
            self.output_dir.mkdir()

        # Build and save (or skip) the file
        if vis_type == "flows":
            filename = f"flows_action_{action_step+1:02d}.png"
        elif vis_type == "vector_field":
            filename = f"vector_field.png"
        else:
            raise ValueError(
                f"Invalid vis_type '{vis_type}'. Expected 'flows' or 'vector_field'."
            )
        filepath = self.output_dir / filename
        if filepath.exists():
            print(f"Warning: File {filepath} already exists. Skipping save.")
        else:
            fig.savefig(filepath, dpi=300)
            print(f"Saved figure to {filepath}.")

    def _create_flows_gif(
        self,
        images_dir: Path,
        action_steps: list | int,
        gif_name: str = "flows_animation.gif",
        duration: float = 0.2,
    ):
        """
        Create an animated GIF from a sequence of saved flow plot images.
        """
        # Build the list of filenames in the right order
        filepaths = [
            images_dir / f"flows_action_{(step+1):02d}.png"
            for step in action_steps
        ]

        # Read each frame
        frames = [imageio.imread(str(fp)) for fp in filepaths]

        # Write out the GIF
        gif_path = images_dir / gif_name
        imageio.mimsave(str(gif_path), frames, duration=duration)
        print(f"Saved GIF to {gif_path}")