import gymnasium as gym
import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from abc import ABC, abstractmethod
from dm_control import mujoco
from matplotlib.collections import LineCollection
from pathlib import Path
from torch import nn, Tensor
from typing import List, Optional, Sequence, Tuple, Union

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
            filename = f"vector_field_{int(time * 100):02d}.png"
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
            **kwargs: Must contain argument `env` which is the Gym environment whose
                rendered RGB frame will be drawn on.
        """
        if "env" not in kwargs:
            raise ValueError(
                "ActionSeqVisualizer expects the keyword argument 'env' (gym.Env), "
                "but it was not provided."
            )
        env: gym.Env = kwargs["env"]
        
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

        # Sample a batch of action sequences
        actions = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(self.num_action_seq, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )

        # Convert the model-predicted actions from the network’s normalised space
        # back into world coordinates
        actions = self.unnormalize_outputs({"action": actions})["action"]

        # Plotting of action sequences depends on environment
        if env.spec.namespace == "gym_aloha":
            fig = self._create_aloha_action_seq_image(env=env, actions=actions.cpu())
        elif env.spec.namespace == "gym_pusht":
            frame = env.render()
            fig = self._create_pusht_action_seq_image(frame=frame, actions=actions.cpu())
        else:
            raise ValueError(
                f"ActionSeqVisualizer does not support environment with namespace '{env.spec.namespace}'."
            )
        
        if self.show:
            plt.show(block=True)
        if self.save:
            self._save_figure(fig)

        plt.close(fig)

    def _project_world_point_to_pixels(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        point_world: np.ndarray,
        camera_name: str = "top",
        img_width: int = 640,
        img_height: int = 480
    ):
        """
        Project a single 3-D point from the world frame into pixel coordinates.
        """
        # Get the camera extrinsics
        cam_id = mujoco.mj_name2id(
            model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, camera_name.encode()
        )
        cam_pos = data.cam_xpos[cam_id]
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

        # Convert point from world to camera coordinates
        p_cam = cam_mat @ (point_world - cam_pos)

        # Everything in front of the camera has negative z in MuJoCo
        if p_cam[2] >= 0:
            raise ValueError("Point is behind the camera")

        # Get the camera intrinsics
        fovy_deg = model.cam_fovy[cam_id]
        fovy_rad = np.deg2rad(fovy_deg)
        f = 0.5 * img_height / np.tan(0.5 * fovy_rad)

        # Get the image centre
        cx, cy = img_width * 0.5, img_height * 0.5

        # Perspective projection
        u = ( p_cam[0] / -p_cam[2] ) * f + cx
        v = (-p_cam[1] / -p_cam[2] ) * f + cy

        return u, v

    def _create_aloha_action_seq_image(self, env: gym.Env, actions: Tensor) -> plt.Figure:
        """
        Render action trajectories on top of a Aloha-Gym environment frame.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()
        
        # Access the underlying MuJoCo physics wrapper
        physics = env.unwrapped._env.physics

        # Save the initial MuJoCo state (qpos + qvel) so we can restore later
        initial_state = physics.get_state()

        # Render the current camera image (RGB), using the "top" camera
        frame = env.render()

        # Prepare to store 3D waypoints for each action sequence
        all_waypoints_left:  List[List[np.ndarray]] = []
        all_waypoints_right: List[List[np.ndarray]] = []

        # For each action sequence: restore initial state, step through the env, and record fingertip midpoints
        for seq_idx in range(actions.shape[0]):
            # Restore the saved state before simulating this sequence
            physics.set_state(initial_state)
            physics.forward()

            seq_waypoints_left: List[np.ndarray] = []
            seq_waypoints_right: List[np.ndarray] = []
            action_seq = actions[seq_idx].cpu().numpy()

            for action_step in range(self.config.horizon):
                env.step(action_seq[action_step])
        
                # After stepping, get fingertip positions for left arm
                left_fingertip_left = physics.data.body("vx300s_left/left_finger_link").xpos
                left_fingertip_right = physics.data.body("vx300s_left/right_finger_link").xpos
                midpoint_left = 0.5 * (left_fingertip_left + left_fingertip_right)
                seq_waypoints_left.append(np.array(midpoint_left))

                # Similarly, get fingertip positions for right arm
                right_fingertip_left = physics.data.body("vx300s_right/left_finger_link").xpos
                right_fingertip_right = physics.data.body("vx300s_right/right_finger_link").xpos
                midpoint_right = 0.5 * (right_fingertip_left + right_fingertip_right)
                seq_waypoints_right.append(np.array(midpoint_right))
                
            all_waypoints_left.append(seq_waypoints_left)
            all_waypoints_right.append(seq_waypoints_right)

        physics.set_state(initial_state)
        physics.forward()
        
        # Create a Matplotlib figure and axis with the same extents as the image
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.imshow(frame)
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # invert y-axis so that v increases downward
        ax.set_aspect("equal")
        ax.axis("off")

        # Prepare a colormap over the trajectory length
        norm = plt.Normalize(0, self.config.horizon - 1)
        cmap = plt.get_cmap("turbo")

        def _draw_waypoints(waypoints: List[np.ndarray]):
            pixel_points: list[Tuple[float, float]] = []
            for point_3d in waypoints:
                try:
                    u_px, v_px = self._project_world_point_to_pixels(
                        physics.model,
                        physics.data,
                        point_3d,
                        camera_name="top",
                        img_width=640,
                        img_height=480
                    )
                    pixel_points.append((u_px, v_px))
                except ValueError:
                    # If a waypoint is behind the camera, skip plotting it
                    continue

            if len(pixel_points) < 2:
                # Cannot form a line segment with fewer than 2 points
                return
            
            # Build line segments between consecutive projected points
            segments = [
                [pixel_points[i], pixel_points[i + 1]]
                for i in range(len(pixel_points) - 1)
            ]
            line_collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=2,
                linestyles="solid",
            )
            # Color each segment by its timestep index
            line_collection.set_array(np.arange(self.config.horizon - 1))
            ax.add_collection(line_collection)

            # Mark the final waypoint with a filled circle
            final_u_px, final_v_px = pixel_points[-1]
            ax.scatter(
                final_u_px,
                final_v_px,
                c=[cmap(1.0)],
                s=30,
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
            )

        # For each sequence, project 3D waypoints to pixel coords and draw
        for waypoints_left, waypoints_right in zip(all_waypoints_left, all_waypoints_right):
            _draw_waypoints(waypoints_left)
            _draw_waypoints(waypoints_right)           

        plt.tight_layout(pad=0)

        if was_interactive:
            plt.ion()

        return fig

    def _create_pusht_action_seq_image(self, frame: np.ndarray, actions: Tensor) -> plt.Figure:
        """
        Render action trajectories on top of a Push-T Gym environment frame.
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
        axis_limits: Optional[Sequence[Tuple[float, float]]],
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
        if not isinstance(action_dims, (list, tuple)) or len(action_dims) not in (2, 3):
            raise ValueError(
                "'action_dims' must be a list or tuple of length 2 or 3, "
                f"but got {action_dims}"
            )

        if axis_limits is not None and len(action_dims) != len(axis_limits):
            raise ValueError(
                f"'axis_limits' length ({len(axis_limits)}) must match 'action_dims' length "
                f"({len(action_dims)})."
            )

        self.action_dims = action_dims
        self.axis_limits = axis_limits
        # Visualize all action steps by default
        if action_steps is None:
            self.action_steps = list(range(self.config.horizon))
        else:
            self.action_steps = action_steps
        self.num_paths = num_paths
        self.vis_type = "flows"

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
        time_grid = torch.linspace(0.0, 1.0, steps=11, device=device)

        # Sample paths from the ODE
        ode_states = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(self.num_paths, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
            time_grid=time_grid,
            return_intermediate_states=True,
        ) # Shape: (timesteps, num_paths, horizon, action_dim)

        # Extract the final sampled action sequences and the flow paths from the ODE states
        final_sample = ode_states[-1] # Shape: (num_paths, horizon, action_dim)
        paths = ode_states[:-1].transpose(0, 1) # Shape: (num_paths, timesteps-1, horizon, action_dim)
        
        # Compute vector field (velocity at each position)
        vector_field = torch.empty_like(paths)
        for p in range(self.num_paths):
            path = paths[p]
            with torch.no_grad():
                path_velocities = self.velocity_model(
                    path,
                    time_grid[:-1],
                    global_cond.repeat(len(time_grid[:-1]), 1)
                )
            vector_field[p] = path_velocities
        
        # Select the specific action step and dimensions
        sample_position = final_sample[..., self.action_dims]
        path_positions = paths[..., self.action_dims]
        velocity_vectors = vector_field[..., self.action_dims]
        
        # Compute global axis limits to create plots of equal size
        if self.axis_limits is None:
            self.axis_limits = []
            for i in range(len(self.action_dims)):
                coords = path_positions[..., i]           # (num_paths, t, steps)
                coord_min, coord_max = coords.min().cpu(), coords.max().cpu()
                margin = 0.05 * (coord_max - coord_min)
                self.axis_limits.append((coord_min - margin, coord_max + margin))

        # Create a separate figure for each action step
        for action_step in self.action_steps:
            sample_position_single_action = sample_position[:, action_step, :]
            path_positions_single_action = path_positions[:, :, action_step, :]
            velocity_vectors_single_action = velocity_vectors[:, :, action_step, :]

            if len(self.action_dims) == 2:
                fig = self._plot_flows_2d(
                    path_positions=path_positions_single_action,
                    velocity_vectors=velocity_vectors_single_action,
                    sample_position=sample_position_single_action,
                    time_grid=time_grid,
                    num_paths=self.num_paths,
                    action_step=action_step,
                )
            else:
                fig = self._plot_flows_3d(
                    path_positions=path_positions_single_action,
                    velocity_vectors=velocity_vectors_single_action,
                    sample_position=sample_position_single_action,
                    time_grid=time_grid,
                    num_paths=self.num_paths,
                    action_step=action_step,
                )

            # Show plot if requested
            if self.show:
                plt.show(block=True)

            # Save plot if requested
            if self.save:
                self._save_figure(fig, action_step=action_step)
        
            plt.close(fig)

        if self.create_gif:
            self._create_gif()

    def _plot_flows_3d(
        self,
        path_positions: Tensor,
        velocity_vectors: Tensor,
        sample_position: Tensor,
        time_grid: Tensor,
        num_paths: int,
        action_step: int,
    ):
        """
        Draw a 3D quiver plot for the flow of a single action step.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        # Extract x-, y- and z-coordinates and velocities
        x, y, z = [path_positions[..., i].flatten().cpu() for i in range(3)]
        u, v, w = [velocity_vectors[..., i].flatten().cpu() for i in range(3)]
        
        # Color arrows by time
        times_grid = time_grid.repeat(num_paths).cpu().numpy()
        times_min, times_max = times_grid.min(), times_grid.max()
        time_norm = (times_grid - times_min) / (times_max - times_min)
        cmap = cm.get_cmap('viridis')
        colors = cmap(time_norm)

        # Scale original vectors by normalised time                                        # overall scale
        length_scale = 0.1
        u_scaled = u * time_norm
        v_scaled = v * time_norm
        w_scaled = w * time_norm

        # Create quiver plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
        fig.canvas.manager.set_window_title("Visualization of Flows")
        quiv = ax.quiver(
            x, y, z,
            u_scaled, v_scaled, w_scaled, 
            length=length_scale,
            linewidth=1.5,
            arrow_length_ratio=0.25,
            normalize=False,
            color=colors
        )

        # Add red dots for sample positions
        sample_x = sample_position[:, 0].cpu()
        sample_y = sample_position[:, 1].cpu()
        sample_z = sample_position[:, 2].cpu()
        ax.scatter(
            sample_x, sample_y, sample_z,
            color='red', s=10, depthshade=True, zorder=3,
            label='Sample Positions'
        )

        # Set consistent axis limits so the plots of all action steps have same size
        ax.set_xlim(*self.axis_limits[0])
        ax.set_ylim(*self.axis_limits[1])
        ax.set_zlim(*self.axis_limits[2])
        ax.set_aspect('equal')

        # Colorbar and title
        cbar = fig.colorbar(quiv, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel('Time', fontsize=12)
        ax.set_title(f"Flow of Action Step {action_step+1} (Horizon: {self.config.horizon})",
                     fontsize=16)

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

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        plt.tight_layout()

        if was_interactive:
            plt.ion()

        return fig
    
    def _plot_flows_2d(
        self,
        path_positions: Tensor,
        velocity_vectors: Tensor,
        sample_position: Tensor,
        time_grid: Tensor,
        num_paths: int,
        action_step: int,
    ) -> plt.Figure:
        """
        Draw a 2D quiver plot for the flow of a single action step.
        """
        was_interactive = plt.isinteractive()
        plt.ioff() 
        
        # Extract x- and y-coordinates and velocities
        x = path_positions[..., 0].flatten().cpu()
        y = path_positions[..., 1].flatten().cpu()
        u = velocity_vectors[..., 0].flatten().cpu()
        v = velocity_vectors[..., 1].flatten().cpu()

        # Create quiver plot
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.manager.set_window_title("Visualization of Flows")
        quiv = ax.quiver(
            x, y, u, v, time_grid[:-1].repeat(num_paths).cpu(),
            angles='xy', scale=len(time_grid[:-1]),
            scale_units='xy', width=0.004, cmap='viridis'
        )

        # Add red dots for sample positions
        sample_x = sample_position[:, 0].cpu()
        sample_y = sample_position[:, 1].cpu()
        ax.scatter(sample_x, sample_y, color='red', s=10, label='Sample Positions', zorder=3)

        # Set consistent axis limits so the plots of all action steps have same size
        ax.set_xlim(*self.axis_limits[0])
        ax.set_ylim(*self.axis_limits[1])
        ax.set_aspect('equal')

        # Colorbar and title
        cbar = fig.colorbar(quiv, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel('Time', fontsize=12)
        ax.set_title(f"Flow of Action Step {action_step+1} (Horizon: {self.config.horizon})",
                     fontsize=16)

        # Axis labels
        if self.action_dim_names:
            x_label = self.action_dim_names[self.action_dims[0]]
            y_label = self.action_dim_names[self.action_dims[1]]
        else:
            x_label = f"Action dimension {self.action_dims[0]}"
            y_label = f"Action dimension {self.action_dims[1]}"
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
        action_dims: Sequence[int],
        action_steps: Optional[Sequence[int]],
        min_action: float,
        max_action: float,
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
            action_dims: Indices of the action-vector dimensions to visualize.
                Must be length 2 (for 2D) or 3 (for 3D).
            action_steps: Randomly choose one of these action steps along the entire action
                sequence horizon to create the vector field visualization for. If None,
                defaults to [0, 1, ..., horizon - 1].
            min_action: Scalar lower bound for each plotted axis.
            max_action: Scalar upper bound for each plotted axis.
            grid_size: Number of grid points per axis.
            time_grid: A sequence of float values in [0.0, 1.0] indicating the time steps
                at which the vector field is evaluated. If None, defaults to [0.0, 0.05, ..., 1.0].
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
        if len(action_dims) not in (2, 3):
            raise ValueError(
                "The vector-field visualisation supports 2D and 3D only, "
                f"(got action_dims = {action_dims}."
            )

        self.action_dims = action_dims
        if action_steps is None:
            self.action_steps = list(range(self.config.horizon))
        else:
            self.action_steps = action_steps
        self.min_action = min_action
        self.max_action = max_action
        self.grid_size = grid_size
        # Default time_grid is list [0.05, 0.1, ..., 1.0]
        self.time_grid = list(np.linspace(0, 1, 31)) if time_grid is None else time_grid
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
        
        self.run_dir = self._update_run_dir()
        
        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        # Initialize ODE solver
        ode_solver = ODESolver(self.velocity_model)
        
        # Sample single noise vector from prior
        noise_sample = torch.randn(
            size=(1, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
        )

        # Sample a single action sequence
        actions = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(1, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )

        # Always visualize at least the cube [-3, +3] as a reasonable range
        # for the Gaussian noise samples
        min_lim = min(self.min_action, -1.0)
        max_lim = max(self.max_action,  1.0)

        # Build a 1-D lin-space once and reuse it for every axis we need
        axis_lin = np.linspace(min_lim, max_lim, self.grid_size)

        # Create the grids
        if len(self.action_dims) == 2:
            x_grid, y_grid = np.meshgrid(axis_lin, axis_lin, indexing="xy")
            x_dim, y_dim = self.action_dims
        else:
            x_grid, y_grid, z_grid = np.meshgrid(axis_lin, axis_lin, axis_lin, indexing="xy")
            x_dim, y_dim, z_dim = self.action_dims

        action_step = random.choice(self.action_steps)
        positions = actions.repeat(x_grid.size, 1, 1)
        positions[:, action_step, x_dim] = torch.tensor(x_grid.ravel(), dtype=dtype, device=device)
        positions[:, action_step, y_dim] = torch.tensor(y_grid.ravel(), dtype=dtype, device=device)
        if len(self.action_dims) == 3:
            positions[:, action_step, z_dim] = torch.tensor(z_grid.ravel(), dtype=dtype, device=device)

        # Build a condition vector tensor whose batch size is the number of grid points
        num_grid_points = positions.shape[0]
        global_cond_batch = global_cond.repeat(num_grid_points, 1)
        
        # Compute the max velocity norm over the timesteps for coloring
        max_velocity_norm = float('-inf')
        for time in reversed(self.time_grid):
            time_batch = torch.full((num_grid_points,), time, device=device, dtype=dtype)
            with torch.no_grad():
                velocities = self.velocity_model(positions, time_batch, global_cond_batch)
            norms = torch.norm(velocities[:, action_step, self.action_dims], dim=1)
            cur_max_velocity_norm = norms.max().item()
            if cur_max_velocity_norm <= max_velocity_norm:
                break
            max_velocity_norm = cur_max_velocity_norm

        for time in self.time_grid:
            time_batch = torch.full((num_grid_points,), time, device=device, dtype=dtype)
            # Compute velocity at grid points and current time as given by flow matching velocity model
            with torch.no_grad():
                velocities = self.velocity_model(positions, time_batch, global_cond_batch)

            if len(self.action_dims) == 2:
                fig = self._create_vector_field_plot_2d(
                    x_positions=x_grid.reshape(-1),
                    y_positions=y_grid.reshape(-1),
                    x_velocities=velocities[:, action_step, x_dim].cpu().numpy(),
                    y_velocities=velocities[:, action_step, y_dim].cpu().numpy(),
                    limits=(min_lim, max_lim),
                    action_step=action_step,
                    time=time,
                    velocity_norm=max_velocity_norm
                )
            else:
                fig = self._create_vector_field_plot_3d(
                    x_positions=x_grid.reshape(-1),
                    y_positions=y_grid.reshape(-1),
                    z_positions=z_grid.reshape(-1),
                    x_velocities=velocities[:, action_step, x_dim].cpu().numpy(),
                    y_velocities=velocities[:, action_step, y_dim].cpu().numpy(),
                    z_velocities=velocities[:, action_step, z_dim].cpu().numpy(),
                    limits=(min_lim, max_lim),
                    action_step=action_step,
                    time=time,
                    velocity_norm=max_velocity_norm
                )

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
        velocity_norm: float,
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
        cmap = cm.get_cmap('viridis')
        colors = cmap(norms / velocity_norm)

        # Create quiver plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
        fig.canvas.manager.set_window_title("Visualization of Vector Field")
        quiv = ax.quiver(
            x_positions, y_positions, z_positions,
            x_velocities, y_velocities, z_velocities,
            length=0.025,
            normalize=False,
            linewidth=0.7,
            arrow_length_ratio=0.2,
            colors=colors
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
        ax.set_title(f"Vector Field of Action Step {action_step} at t={time:.2f}", fontsize=16)

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
        velocity_norm: float,
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
        cmap = cm.get_cmap('viridis')
        colors = cmap(norms / velocity_norm)
        
        # Create quiver plot
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.manager.set_window_title("Visualization of Vector Field")
        quiv = ax.quiver(
            x_positions, y_positions,
            x_velocities, y_velocities,
            angles='xy', scale=40,
            scale_units='xy', width=0.004,
            color=colors
        )

        # Set axis limits
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')

        # Colorbar and title
        cbar = fig.colorbar(quiv, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel('Velocity Norm', fontsize=12)
        
        # Title
        ax.set_title(f"Vector Field of Action Step {action_step} at t={time:.2f}", fontsize=16)

        # Axis labels
        if self.action_dim_names:
            x_label = self.action_dim_names[self.action_dims[0]]
            y_label = self.action_dim_names[self.action_dims[1]]
        else:
            x_label = f"Action dimension {self.action_dims[0]}"
            y_label = f"Action dimension {self.action_dims[1]}"
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        plt.tight_layout()
        
        if was_interactive:
            plt.ion()

        return fig