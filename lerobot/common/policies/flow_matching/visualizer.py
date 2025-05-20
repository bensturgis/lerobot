import imageio
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from torch import nn, Tensor
from typing import Optional, Sequence, Tuple, Union

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.ode_solver import ODESolver
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

class FlowMatchingVisualizer(ABC):
    """
    Abstract base class for flow matching visualizer.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        action_dim_names: Optional[Sequence[str]],
        show: bool,
        save: bool,
        output_root: Optional[Union[Path, str]],
        create_gif: bool,
        verbose: bool,
    ):
        """
        Args:
            config: Configuration object for Flow Matching settings.
            velocity_model: The learned flow matching velocity model.
            action_dim_names: Optional names for action dimensions (used for axis labels).
            show: If True, display the plots.
            save: If True, save the plots to disk.
            output_root: Optional output directory for saving figures.
            create_gif: If True, create a GIF from the saved figures.
            verbose: If True, print status messages.
        """
        self.config = config
        self.velocity_model = velocity_model
        self.action_dim_names = action_dim_names
        self.show = show
        self.save = save
        self.output_root = output_root
        self.create_gif = create_gif
        self.verbose = verbose

    @abstractmethod
    def visualize(self, global_cond: Tensor, **kwargs):
        """
        Run the visualization using the provided conditioning vector.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
            Shape [cond_dim,] or [1, cond_dim].
            **kwargs: Visualiser-specific keyword arguments.
        """
        pass

    def _update_run_dir(self) -> Path:
        """
        Create a new, empty folder and return its path.
        """
        if self.output_root is None:
            self.output_root = Path("outputs/visualizations/")
        
        vis_type_dir = self.output_root / self.vis_type
        vis_type_dir.mkdir(parents=True, exist_ok=True)
        if self.vis_type == "action_seq":
            return vis_type_dir
        else:
            run_idx = 1
            while (vis_type_dir / f"{run_idx:04d}").exists():
                run_idx += 1
            run_dir = vis_type_dir / f"{run_idx:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
    
    def _save_figure(
        self,
        fig: plt.Figure,
        action_step: Optional[int] =  None,
        time: Optional[float] = None,
    ):
        """
        Save the given figure to a visualization directory based on the visualization type.
        """
        if self.vis_type == "flows" and action_step is None:
            raise ValueError("`action_step` must be provided when visualization type is 'flows'.")
        if self.vis_type == "vector_field" and time is None:
            raise ValueError("`time` must be provided when visualization type is 'vector_field'.")

        # Build and save (or skip) the file
        if self.vis_type == "flows":
            filename = f"flows_action_{action_step+1:02d}.png"
        elif self.vis_type == "vector_field":
            filename = f"vector_field_{int(time * 10):02d}.png"
        elif self.vis_type == "action_seq":
            action_seq_idx = 1
            while (self.run_dir / f"action_seqs_{action_seq_idx:04d}.png").exists():
                action_seq_idx += 1
            filename = f"action_seqs_{action_seq_idx:04d}"
        else:
            raise ValueError(
                f"Invalid vis_type '{self.vis_type}'. Expected 'flows' or 'vector_field'."
            )
        filepath = self.run_dir / filename
        if filepath.exists():
            if self.verbose:
                print(f"Warning: File {filepath} already exists. Skipping save.")
        else:
            fig.savefig(filepath, dpi=300)
            if self.verbose:
                print(f"Saved figure to {filepath}.")

    def _create_gif(self, duration: float = 0.2):
        """
        Create an animated GIF from a sequence of saved flow
        or vector field plot images.
        """
        # Build the list of filenames in the right order
        filepaths = sorted([fp for fp in self.run_dir.iterdir() if fp.suffix.lower() == ".png"])

        # Read each frame
        frames = [imageio.imread(str(fp)) for fp in filepaths]

        # Write out the GIF
        if self.vis_type == "flows":
            gif_name = "flow_animation.gif"
        elif self.vis_type == "vector_field":
            gif_name = "vector_field_animation.gif"
        else:
            raise ValueError(
                f"Invalid vis_type '{self.vis_type}'. Expected 'flows' or 'vector_field'."
            )
        gif_path = self.run_dir / gif_name
        if gif_path.exists():
            if self.verbose:
                print(f"Warning: File {gif_path} already exists. Skipping save.")
        else:
            imageio.mimsave(str(gif_path), frames, duration=duration)
            if self.verbose:
                print(f"Saved GIF to {gif_path}")


# TODO: Add action sequence visualization to live stream and video of policy rollout
class ActionSeqVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting a batch of action sequences onto the current observation frame.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        unnormalize_outputs: nn.Module,
        num_action_seq: int,
        action_dim_names: Optional[Sequence[str]],
        output_root: Optional[Union[Path, str]],
        show: bool = False,
        save: bool = True,
        create_gif: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            unnormalize_outputs: Module to map model outputs from normalized space
                back to original action coordinates.
            num_action_seq: Number of action sequences to plot.
        """
        super().__init__(
            config=config,
            velocity_model=velocity_model,
            action_dim_names=action_dim_names,
            show=show,
            save=save,
            output_root=output_root,
            create_gif=create_gif,
            verbose=verbose,
        )
        if self.config.action_feature.shape[0] != 2:
            raise ValueError(
                "The action sequence visualisation requires action_dim = 2, "
                f"but got action_dim = {self.config.action_feature.shape[0]}."
            )
        self.num_action_seq = num_action_seq
        self.unnormalize_outputs = unnormalize_outputs
        self.vis_type = "action_seq"
        self.run_dir = self._update_run_dir()
        
    def visualize(self, global_cond: Tensor, **kwargs):
        """
        Visualize a batch of action sequences onto the current frame.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape: [cond_dim,] or [1, cond_dim].
            **kwargs: Must contain argument `frame` which is the current RGB video
                frame to draw on. Shape: [H, W, 3].
        """
        if "frame" not in kwargs:
            raise ValueError(
                "ActionSeqVisualizer expects the keyword argument 'frame' (RGB image), "
                "but it was not provided."
            )
        frame: np.ndarray = kwargs["frame"]
        
        if global_cond.dim() == 1: # shape = (cond_dim,)
            global_cond = global_cond.unsqueeze(0)     # (1, cond_dim)
        elif global_cond.dim() == 2 and global_cond.size(0) == 1: # shape = (1, cond_dim)
            pass
        else:
            raise ValueError(
                f"Expected global_cond to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)
        
        # Initialize ODE solver
        ode_solver = ODESolver(self.velocity_model)
        
        # Sample noise from prior
        noise_sample = torch.randn(
            size=(self.num_action_seq, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
        )

        # Sample a bathc of action sequences
        actions = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(self.num_action_seq, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )

        # Convert the model-predicted actions from the network’s normalised space
        # back into PushT world coordinates
        actions = self.unnormalize_outputs({"action": actions})["action"]

        fig = self._create_action_seq_image(frame=frame, actions=actions.cpu())

        if self.show:
            plt.show(block=True)
        if self.save:
            self._save_figure(fig)

        plt.close(fig)

    def _create_action_seq_image(self, frame: np.ndarray, actions: Tensor) -> plt.Figure:
        """
        Render action trajectories on top of a Gym environment frame.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        # Flip the image vertically to convert from image to world coords
        frame = np.flipud(frame)

        # Create a square figure that maps world coords [0,512] to the image axes
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        ax.imshow(frame, extent=[0, 512, 0, 512])
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_aspect("equal")
        ax.axis("off")
    
        # Prepare a colormap over the trajectory length
        norm = plt.Normalize(0, self.config.horizon - 1)
        cmap = plt.get_cmap("turbo")
    
        # Draw each action sequence as a colourful line strip
        for action_seq in actions:
            segments = [
                [tuple(action_seq[i]), tuple(action_seq[i+1])]
                for i in range(self.config.horizon - 1)
            ]
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=2,
                linestyles='solid',
            )
            # Assign each segment its action step index as the “value”
            # for the colour mapping
            lc.set_array(np.arange(self.config.horizon - 1))
            ax.add_collection(lc)

            # Mark the final action target with a filled circle
            ax.scatter(
                action_seq[-1, 0], action_seq[-1, 1],
                c=[cmap(1.0)], s=30, edgecolors='k',
                linewidths=0.5, zorder=3,
            )

        plt.tight_layout(pad=0)

        if was_interactive:
            plt.ion()

        return fig


class FlowVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting flow trajectories through the learned velocity field.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        action_dims: Sequence[int],
        action_steps: Optional[Union[Sequence[int], int]],
        num_paths: int,
        action_dim_names: Optional[Sequence[str]],
        output_root: Optional[Union[Path, str]],
        show: bool = False,
        save: bool = True,
        create_gif: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            action_dims: Two action dimensions to plot.
            action_steps: Steps to visualize; defaults to all.
            num_paths: Number of trajectories per step.
        """
        super().__init__(
            config=config,
            velocity_model=velocity_model,
            action_dim_names=action_dim_names,
            show=show,
            save=save,
            output_root=output_root,
            create_gif=create_gif,
            verbose=verbose,
        )
        if not isinstance(action_dims, (list, tuple)) or len(action_dims) != 2:
            raise ValueError(
                "'action_dims' must be a list or tuple of exactly two elements, "
                f"but got {action_dims}"
            )

        self.action_dims = action_dims
        # Visualize all action steps by default
        if action_steps is None:
            self.action_steps = list(range(self.config.horizon))
        else:
            self.action_steps = action_steps
        self.num_paths = num_paths
        self.vis_type = "flows"

    # TODO:
    # - Add option to visualize 3D flows
    def visualize(self, global_cond: Tensor, **kwargs):
        """
        Visualize flow trajectories for specified action steps and dimensions.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape [cond_dim,] or [1, cond_dim].
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

        self.run_dir = self._update_run_dir()

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)
        
        # Initialize ODE solver
        ode_solver = ODESolver(self.velocity_model)
        
        # Sample noise from prior
        noise_sample = torch.randn(
            size=(self.num_paths, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
        )
        
        # Create time grid
        time_grid = torch.arange(0.0, 1.0, 0.05, device=device)

        # Sample paths from the ODE
        paths = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(self.num_paths, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
            time_grid=time_grid,
            return_intermediates=True,
        ).transpose(0, 1)  # Shape: (num_paths, timesteps, horizon, action_dim)

        # Compute vector field (velocity at each position)
        vector_field = torch.empty_like(paths)
        for p in range(self.num_paths):
            path = paths[p]
            with torch.no_grad():
                path_velocities = self.velocity_model(
                    path,
                    time_grid,
                    global_cond.repeat(len(time_grid), 1)
                )
            vector_field[p] = path_velocities
        
        # Select the specific action step and dimensions
        positions = paths[..., self.action_dims]
        velocity_vectors = vector_field[..., self.action_dims]
        
        # Compute global axis limits to create plots of equal size
        x_positions, y_positions = positions[:, :, self.action_steps, 0], positions[:, :, self.action_steps, 1]
        x_min, x_max = x_positions.min().cpu(), x_positions.max().cpu()
        y_min, y_max = y_positions.min().cpu(), y_positions.max().cpu()
        margin_x = 0.05 * (x_max - x_min)
        margin_y = 0.05 * (y_max - y_min)
        x_lim = (x_min - margin_x, x_max + margin_x)
        y_lim = (y_min - margin_y, y_max + margin_y)

        # Create a separate figure for each action steo
        for action_step in self.action_steps:
            positions_single_action = positions[:, :, action_step, :]
            velocity_vectors_single_action = velocity_vectors[:, :, action_step, :]

            fig = self._create_flow_plot(
                positions_single_action, velocity_vectors_single_action,
                time_grid, self.num_paths,
                x_lim, y_lim,
                action_step, self.action_dims
            )

            # Show plot if requested
            if self.show:
                fig.show(block=True)

            # Save plot if requested
            if self.save:
                self._save_figure(fig, action_step=action_step)
        
            plt.close(fig)

        if self.create_gif:
            self._create_gif()

    def _create_flow_plot(
        self,
        positions: Tensor,
        velocity_vectors: Tensor,
        time_grid: Tensor,
        num_paths: int,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        action_step: int,
        action_dims: Sequence[int] = (0, 1),
    ) -> plt.Figure:
        """
        Draw a quiver plot for the flow of a single action step.
        """
        was_interactive = plt.isinteractive()
        plt.ioff() 
        
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
        ax.set_title(f"Flow of Action Step {action_step+1} (Horizon: {self.config.horizon})",
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

        if was_interactive:
            plt.ion()
        
        return fig
    

class VectorFieldVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting vector fields.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        min_action: np.array,
        max_action: np.array,
        grid_size: int,
        time_grid: Optional[Sequence[float]],
        action_dim_names: Optional[Sequence[str]],
        output_root: Optional[Union[Path, str]],
        show: bool = False,
        save: bool = True,
        create_gif: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            min_action: Lower bounds for x and y (shape (2,)).
            max_action: Upper bounds for x and y (shape (2,)).
            grid_size: Number of grid points per axis.
            time_grid: A sequence of float values in [0.0, 1.0] indicating the time steps
                   at which the vector field is evaluated. If None, defaults to
                   [0.0, 0.1, ..., 1.0].
        """
        super().__init__(
            config=config,
            velocity_model=velocity_model,
            action_dim_names=action_dim_names,
            show=show,
            save=save,
            output_root=output_root,
            create_gif=create_gif,
            verbose=verbose,
        )
        if self.config.action_feature.shape[0] != 2 or self.config.horizon != 1:
            raise ValueError(
                "The vector-field visualisation requires action_dim = 2 and horizon = 1 "
                f"(got action_dim = {self.config.action_feature.shape[0]}, "
                f"horizon = {self.config.horizon})."
            )

        self.min_action = min_action
        self.max_action = max_action
        self.grid_size = grid_size
        # Default time_grid is list [0.1, 0.2, ..., 1.0]
        if self.time_grid is None:
            self.time_grid = list(np.linspace(0, 1, 11))
        else:
            self.time_grid = time_grid
        self.vis_type = "vector_field"
    
    def visualize(self, global_cond: Tensor, **kwargs):
        """
        Visualize the 2D action vector field produced by a flow matching policy at a given time.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape [cond_dim,] or [1, cond_dim].
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
        
        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        self.run_dir = self._update_run_dir()

        # Clamp values to [-3, +3]
        min_lim = np.minimum(self.min_action, -3.0)
        max_lim = np.maximum(self.max_action,  3.0)

        # Unpack minimum x- and y-coordinates
        x_min, y_min = min_lim
        x_max, y_max = max_lim

        # Create a 2D meshgrid in the range of [x_min, x_max] x [y_min, y_max]
        x_lin = np.linspace(x_min, x_max, self.grid_size)
        y_lin = np.linspace(y_min, y_max, self.grid_size)
        x_grid, y_grid = np.meshgrid(x_lin, y_lin, indexing="xy")
        
        # Flatten numpy grids and turn them into a (num_grid_points,1,2) torch tensor to input into
        # the velocity model
        positions = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=-1)
        positions = torch.from_numpy(positions).unsqueeze(1).to(device=device, dtype=dtype)

        # Build a condition vector tensor whose batch size is the number of grid points
        num_grid_points = positions.shape[0]
        global_cond_batch = global_cond.repeat(num_grid_points, 1)

        
        for time in self.time_grid:
            time_batch = torch.full((num_grid_points,), time, device=device, dtype=dtype)
            # Compute velocity at grid points and current time as given by flow matching velocity model
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

            if self.show:
                plt.show(block=True)
            if self.save:
                self._save_figure(fig, time=time)

            plt.close(fig)

        if self.create_gif:
            self._create_gif()
    
    def _create_vector_field_plot(
        self,
        x_positions: np.array,
        y_positions: np.array,
        x_velocities: np.array,
        y_velocities: np.array,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        time: float,
    ) -> plt.Figure:
        """
        Draw a quiver plot for the vector field of a single two-dimensional action at a
        given time.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()
        
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
        
        if was_interactive:
            plt.ion()

        return fig