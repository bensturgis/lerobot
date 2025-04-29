import imageio
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch import nn, Tensor

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.ode_solver import ODESolver
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

# TODO:
# - Implement visualize_vector_field() method
# - Retrain PushT Flow Matching Policy for H=1
# - Retrain PushT Flow Matching Policy for scaled environment to better visualize flows

class FlowMatchingVisualizer:
    """
    Visualizer for Flow Matching paths and vector fields of action sequences.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        global_cond: Tensor,
        action_dim_names: list | tuple | None = None,
        output_dir: Path | str | None = None
    ):
        """
        Args:
            config: Configuration object for Flow Matching settings.
            velocity_model: The learned velocity model.
            global_cond: Conditioning tensor for the model.
            action_dim_names: Optional names for action dimensions (used for axis labels).
            output_dir: Optional output directory for saving figures.
        """
        self.config = config
        self.velocity_model = velocity_model
        self.global_cond = global_cond
        self.action_dim_names = action_dim_names
        self.output_dir = output_dir

    # TODO:
    # - Add option to visualize 3D flows
    def visualize_flows(
        self,
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
            action_dims: Two action dimensions to plot.
            action_steps: Steps to visualize; defaults to all.
            num_paths: Number of trajectories per step.
            show: If True, display the plots.
            save: If True, save the plots to disk.
        """
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
            global_cond=self.global_cond.repeat(num_paths, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
            time_grid=time_grid,
            return_intermediates=True,
        ).transpose(0, 1)  # Shape: (num_paths, horizon, action_dim)

        # Compute vector field (velocity at each position)
        vector_field = torch.empty_like(paths)
        for p in range(num_paths):
            path = paths[p]
            with torch.no_grad():
                path_velocities = self.velocity_model(
                    path,
                    time_grid,
                    self.global_cond.repeat(len(time_grid), 1)
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
                fig.show()

            # Save plot if requested
            if save:
                self._save_flow_figure(fig, action_step)
        
        if create_gif:
            self._create_flows_gif(
                images_dir=self.output_dir,
                action_steps=action_steps,
            )

    

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
    
    def _save_flow_figure(self, fig: plt.Figure, action_step: int) -> None:
        """
        Save the given figure as 'flows_action_##.png', skipping if it already exists.
        """
        # Create a new run folder if needed
        if self.output_dir is None:
            base_dir = Path("outputs/visualizations/flows")
            base_dir.mkdir(parents=True, exist_ok=True)
            run_idx = 1
            while (base_dir / f"{run_idx:04d}").exists():
                run_idx += 1
            self.output_dir = base_dir / f"{run_idx:04d}"
            self.output_dir.mkdir()

        # Build and save (or skip) the file
        filename = f"flows_action_{action_step+1:02d}.png"
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