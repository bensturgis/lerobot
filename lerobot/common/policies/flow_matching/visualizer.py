import gymnasium as gym
import imageio
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from abc import ABC, abstractmethod
from dm_control import mujoco
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from pathlib import Path
from torch import nn, Tensor
from typing import Dict, List, Optional, Sequence, Tuple, Union

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
)
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.configs.default import (
    ActionSeqVisConfig,
    FlowVisConfig,
    VectorFieldVisConfig,
)


class FlowMatchingVisualizer(ABC):
    """
    Abstract base class for flow matching visualizer.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        velocity_model: nn.Module,
        save: bool,
        output_root: Optional[Union[Path, str]],
        create_gif: bool,
        verbose: bool,
    ):
        """
        Args:
            flow_matching_cfg: Configuration object for Flow Matching settings.
            velocity_model: The learned flow matching velocity model.
            show: If True, display the plots.
            save: If True, save the plots to disk.
            output_root: Optional output directory for saving figures.
            create_gif: If True, create a GIF from the saved figures.
            verbose: If True, print status messages.
        """
        self.flow_matching_cfg = flow_matching_cfg
        self.velocity_model = velocity_model
        self.save = save
        self.output_root = output_root
        self.create_gif = create_gif
        self.verbose = verbose

    @abstractmethod
    def visualize(
        self, global_cond: Tensor, generator: Optional[torch.Generator], **kwargs
    ):
        """
        Run the visualization using the provided conditioning vector.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
            Shape [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.
            **kwargs: Visualiser-specific keyword arguments.
        """
        pass

    def _update_run_dir(
        self, vis_type_dir_name: Optional[str] = None
    ) -> Path:
        """
        Create a new, empty folder and return its path.
        """
        if self.output_root is None:
            self.output_root = Path("outputs/visualizations/")
        
        if vis_type_dir_name is None:
            vis_type_dir = self.output_root / self.vis_type
        else:
            vis_type_dir = self.output_root / vis_type_dir_name
        vis_type_dir.mkdir(parents=True, exist_ok=True)
        if self.vis_type == "action_seq":
            return vis_type_dir
        else:
            run_idx = 1
            while (vis_type_dir / f"{run_idx:03d}").exists():
                run_idx += 1
            run_dir = vis_type_dir / f"{run_idx:03d}"
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
            filename = f"vector_field_{int(time * 100):03d}.png"
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
            # Presentation
            # fig.savefig(filepath, dpi=600)
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

        # Get the path where the GIF will be saved
        vis_type_dir = self.output_root / self.vis_type

        # Get the path of the GIF file based on the visualization type and run index
        if self.vis_type == "flows":
            gif_name_base = "flow_animation"
        elif self.vis_type == "vector_field":
            gif_name_base = "vector_field_animation"
        else:
            raise ValueError(
                f"Invalid vis_type '{self.vis_type}'. Expected 'flows' or 'vector_field'."
            )

        run_idx = 1
        while (vis_type_dir / f"{gif_name_base}_{run_idx:03d}.gif").exists():
            run_idx += 1
        gif_path = vis_type_dir / f"{gif_name_base}_{run_idx:03d}.gif"

        # Save the GIF
        imageio.mimsave(str(gif_path), frames, duration=duration)
        if self.verbose:
            print(f"Saved GIF to {gif_path}")

    def _add_actions(
        self,
        ax: Axes,
        action_data: Dict[str, Tensor],
        action_step: int,
        colors: Optional[List[str]] = None,
        zorder: int = 3,
        scale: float = 10,
    ) -> plt.Figure:
        """
        Overlay action samples on a vector field plot.
        """
        if colors is None:
            colors = ["red", "orange", "green", "purple", "magenta", "brown"]
        
        # Figure out which dims to plot
        if len(self.action_dims) == 2:
            x_dim, y_dim = self.action_dims
            is_3d = False
        else:
            x_dim, y_dim, z_dim = self.action_dims
            is_3d = True

        # Plot each action sequence
        for idx, (name, actions) in enumerate(action_data.items()):
            color = colors[idx % len(colors)]
            # Actions shape: (num_samples, horizon, action_dim)
            x_positions = actions[:, action_step, x_dim].cpu().numpy()
            y_positions = actions[:, action_step, y_dim].cpu().numpy()
            label = name.replace("_", " ")
            if is_3d:
                z_positions = actions[:, action_step, z_dim].cpu().numpy()
                # Presentation s=30
                ax.scatter(
                    x_positions, y_positions, z_positions, label=label,
                    color=color, s=scale, zorder=zorder,
                )
            else:
                # Presentation s=30
                ax.scatter(
                    x_positions, y_positions, label=label,
                    color=color, s=scale, zorder=zorder,    
                )

        # Draw legend using the prettified names
        # Presentation: fontsize=28
        ax.legend()
        return ax.get_figure()


# TODO: Add action sequence visualization to live stream and video of policy rollout
class ActionSeqVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting a batch of action sequences onto the current observation frame.
    """
    def __init__(
        self,
        cfg: ActionSeqVisConfig,
        flow_matching_cfg: FlowMatchingConfig,
        velocity_model: nn.Module,
        unnormalize_outputs: nn.Module,
        output_root: Optional[Union[Path, str]],
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
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            save=save,
            output_root=output_root,
            create_gif=create_gif,
            verbose=verbose,
        )
        self.num_action_seq = cfg.num_action_seq
        self.unnormalize_outputs = unnormalize_outputs
        self.show = cfg.show
        self.vis_type = "action_seq"
        
    def visualize(
        self, global_cond: Tensor, generator: Optional[torch.Generator] = None, **kwargs
    ):
        """
        Visualize a batch of action sequences onto the current frame.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape: [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.
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
        
        dir_name = kwargs.get("dir_name", None)
        self.run_dir = self._update_run_dir(vis_type_dir_name=dir_name)

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)
        
        # Initialize ODE solver
        ode_solver = ODESolver(self.velocity_model)
        
        # Sample noise from prior
        noise_sample = torch.randn(
            size=(self.num_action_seq, self.flow_matching_cfg.horizon, self.flow_matching_cfg.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Sample a batch of action sequences
        actions = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(self.num_action_seq, 1),
            step_size=self.flow_matching_cfg.ode_step_size,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
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
        elif env.spec.namespace == "gym_libero":
            fig = self._create_libero_action_seq_image(env=env, actions=actions.cpu())
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
        camera_name: str,
        img_width: int,
        img_height: int
    ):
        """
        Project a single 3-D point from the world frame into pixel coordinates.
        """
        # Get the camera extrinsics
        cam_id = mujoco.mj_name2id(
            model._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        cam_pos = data.cam_xpos[cam_id]
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

        # Convert point from world to camera coordinates
        p_cam = cam_mat.T @ (point_world - cam_pos)

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
    
    def _draw_waypoints(self, env: gym.Env, waypoints: List[np.ndarray], ax: Axes):
        # Prepare a colormap over the trajectory length
        norm = plt.Normalize(0, self.flow_matching_cfg.horizon - 1)
        cmap = plt.get_cmap("turbo")
        
        pixel_points: List[Tuple[float, float]] = []
        for point_3d in waypoints:
            try:
                u_px, v_px = self._project_world_point_to_pixels(
                    env.unwrapped.sim.model,
                    env.unwrapped.sim.data,
                    point_3d,
                    camera_name="frontview",
                    img_width=256,
                    img_height=256
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
        line_collection.set_array(np.arange(self.flow_matching_cfg.horizon - 1))
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

            for action_step in range(self.flow_matching_cfg.horizon):
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

        # For each sequence, project 3D waypoints to pixel coords and draw
        for waypoints_left, waypoints_right in zip(all_waypoints_left, all_waypoints_right):
            self._draw_waypoints(env=env, waypoints=waypoints_left, ax=ax)
            self._draw_waypoints(env=env, waypoints=waypoints_right, ax=ax)           

        plt.tight_layout(pad=0)

        if was_interactive:
            plt.ion()

        return fig

    def _create_libero_action_seq_image(self, env: gym.Env, actions: Tensor) -> plt.Figure:
        """
        Render action trajectories on top of a LIBERO Gym environment frame.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        # Save the initial MuJoCo state (qpos + qvel) so we can restore later
        initial_state = env.unwrapped.get_sim_state()

        # Render the current camera image (RGB), using the "top" camera
        frame = env.unwrapped.render(camera_name="frontview")

        # Tell the LIBERO/robosuite wrapper to ignore the “episode done” flag,
        # so we can keep stepping through from the same sim state without triggering a reset.
        env.unwrapped.env.ignore_done = True

        # Prepare to store 3D waypoints for each action sequence
        all_waypoints: List[List[np.ndarray]] = []

        for seq_idx in range(actions.shape[0]):
            # Restore the saved state before simulating this sequence
            env.unwrapped.set_state(initial_state)
            env.unwrapped.sim.forward()
            env.unwrapped.check_success()
            env.unwrapped.post_process()
            env.unwrapped.update_observables(force=True)
            
            seq_waypoints: List[np.ndarray] = []

            action_seq = actions[seq_idx].cpu().numpy()
            for action_step in range(self.flow_matching_cfg.horizon):
                obs, _, _, _, _ = env.unwrapped.step(action_seq[action_step])
                end_effector_pos = obs["agent_pos"][:3]
                seq_waypoints.append(end_effector_pos)

            all_waypoints.append(seq_waypoints)

        env.unwrapped.set_state(initial_state)
        env.unwrapped.sim.forward()
        env.unwrapped.check_success()
        env.unwrapped.post_process()
        env.unwrapped.update_observables(force=True)

        # Create a Matplotlib figure and axis with the same extents as the image
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax.imshow(frame)
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)  # invert y-axis so that v increases downward
        ax.set_aspect("equal")
        ax.axis("off")

        # For each sequence, project 3D waypoints to pixel coords and draw
        for waypoints in all_waypoints:
            self._draw_waypoints(env=env, waypoints=waypoints, ax=ax)

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
        norm = plt.Normalize(0, self.flow_matching_cfg.horizon - 1)
        cmap = plt.get_cmap("turbo")
    
        # Draw each action sequence as a colourful line strip
        for action_seq in actions:
            segments = [
                [tuple(action_seq[i]), tuple(action_seq[i+1])]
                for i in range(self.flow_matching_cfg.horizon - 1)
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
            lc.set_array(np.arange(self.flow_matching_cfg.horizon - 1))
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
        cfg: FlowVisConfig,
        flow_matching_cfg: FlowMatchingConfig,
        velocity_model: nn.Module,
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
        if not isinstance(cfg.action_dims, (list, tuple)) or len(cfg.action_dims) not in (2, 3):
            raise ValueError(
                "'action_dims' must be a list or tuple of length 2 or 3, "
                f"but got {cfg.action_dims}"
            )

        if cfg.axis_limits is not None and len(cfg.action_dims) != len(cfg.axis_limits):
            raise ValueError(
                f"'axis_limits' length ({len(cfg.axis_limits)}) must match 'action_dims' length "
                f"({len(cfg.action_dims)})."
            )

        self.action_dims = cfg.action_dims
        self.action_dim_names = cfg.action_dim_names
        self.axis_limits = cfg.axis_limits
        # Visualize all action steps by default
        if cfg.action_steps is None:
            self.action_steps = list(range(self.flow_matching_cfg.horizon))
        else:
            self.action_steps = cfg.action_steps
        self.num_paths = cfg.num_paths
        self.show = cfg.show
        self.vis_type = "flows"

    def _prepare_global_cond(self, global_cond: Tensor) -> Tensor:
        """
        Prepare the global conditioning vector for the velocity model.
        Ensures it has shape (1, cond_dim).
        """
        if global_cond.dim() == 1:  # shape = (cond_dim,)
            global_cond = global_cond.unsqueeze(0)  # (1, cond_dim)
        elif global_cond.dim() == 2 and global_cond.size(0) == 1:  # shape = (1, cond_dim)
            pass
        else:
            raise ValueError(
                f"Expected global_cond to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1, cond_dim)), but got shape {tuple(global_cond.shape)}"
            )
        
        return global_cond

    def _compute_vector_field(
        self, velocity_model: nn.Module, global_cond: Tensor, paths: Tensor, eval_times: Tensor, 
    ) -> Tensor:
        """
        Compute the model’s velocity vectors along each trajectory.
        """
        # Compute velocity at each position
        vector_field = torch.empty_like(paths)
        for p in range(self.num_paths):
            path = paths[p]
            with torch.no_grad():
                path_velocities = velocity_model(
                    path,
                    eval_times,
                    global_cond.repeat(len(eval_times), 1)
                )
            vector_field[p] = path_velocities

        return vector_field
    
    def _compute_axis_limits(self, path_positions: Tensor) :
        self.axis_limits = []
        for i in range(len(self.action_dims)):
            coords = path_positions[..., i]           # (num_paths, t, steps)
            coord_min, coord_max = coords.min().cpu(), coords.max().cpu()
            margin = 0.05 * (coord_max - coord_min)
            self.axis_limits.append((coord_min - margin, coord_max + margin))

    def visualize(
        self, global_cond: Tensor, generator: Optional[torch.Generator] = None, **kwargs
    ):
        """
        Visualize flow trajectories for specified action steps and dimensions.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.
        """
        self.run_dir = self._update_run_dir()

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        global_cond = self._prepare_global_cond(global_cond)       
        
        # Initialize ODE solver
        ode_solver = ODESolver(self.velocity_model)
        
        # Sample noise from prior
        noise_sample = torch.randn(
            size=(self.num_paths, self.flow_matching_cfg.horizon, self.flow_matching_cfg.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator
        )
        
        # Create time grid
        velocity_eval_times = torch.linspace(0.0, 0.9, steps=10, device=device)           

        if self.flow_matching_cfg.ode_solver_method in FIXED_STEP_SOLVERS:
            sampling_time_grid = ode_solver.make_sampling_time_grid(
                step_size=self.flow_matching_cfg.ode_step_size,
                extra_times=velocity_eval_times,
                device=device,
            )
        elif self.flow_matching_cfg.ode_solver_method in ADAPTIVE_SOLVERS:
            sampling_time_grid, _ = torch.sort(torch.unique(velocity_eval_times)).to(device)
            # Append 1.0 if not already there
            if sampling_time_grid[-1].item() != 1.0:
                sampling_time_grid = torch.cat(
                    [sampling_time_grid, torch.tensor([1.0], device=device)]
                )

        # Sample paths from the ODE
        ode_states = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(self.num_paths, 1),
            step_size=self.flow_matching_cfg.ode_step_size,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=sampling_time_grid,
            return_intermediate_states=True,
        ) # Shape: (timesteps, num_paths, horizon, action_dim)

        # Extract the final sampled action sequence
        final_sample = ode_states[-1] # Shape: (num_paths, horizon, action_dim)

        # Extract the flow paths from the ODE states
        eval_ode_states, velocity_eval_times = ode_solver.select_ode_states(
            time_grid=sampling_time_grid,
            ode_states=ode_states,
            requested_times=velocity_eval_times,
        )
        paths = eval_ode_states.transpose(0, 1) # Shape: (num_paths, timesteps-1, horizon, action_dim)
        
        # Compute velocity at each position
        vector_field = self._compute_vector_field(
            velocity_model=self.velocity_model,
            global_cond=global_cond,
            paths=paths,
            eval_times=velocity_eval_times,
        )
        
        # Select the specific action step and dimensions
        path_positions = paths[..., self.action_dims]
        velocity_vectors = vector_field[..., self.action_dims]
        
        # Compute global axis limits to create plots of equal size
        if self.axis_limits is None:
            self._compute_axis_limits(path_positions=path_positions)

        # Create a separate figure for each action step
        for action_step in self.action_steps:
            path_positions_single_action = path_positions[:, :, action_step, :]
            velocity_vectors_single_action = velocity_vectors[:, :, action_step, :]
            vector_fields = {
                "Velocities": velocity_vectors_single_action
            }

            if len(self.action_dims) == 2:
                fig, _ = plt.subplots(figsize=(12, 10))
                is_3d = False
            else:
                fig, _ = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
                is_3d = True

            self._add_actions(
                ax=fig.axes[0],
                action_data={"Sampler ODE States": ode_states.flatten(0,1)},
                action_step=action_step,
                colors=["grey"],
                scale=5,
                zorder=1
            )

            if not is_3d:
                fig = self._plot_flows_2d(
                    fig=fig,
                    path_positions=path_positions_single_action,
                    vector_fields=vector_fields,
                    time_grid=velocity_eval_times,
                    num_paths=self.num_paths,
                    action_step=action_step,
                )
            else:
                fig = self._plot_flows_3d(
                    fig=fig,
                    path_positions=path_positions_single_action,
                    velocity_vectors=velocity_vectors_single_action,
                    vector_fields=vector_fields,
                    time_grid=velocity_eval_times,
                    num_paths=self.num_paths,
                    action_step=action_step,
                )

            # Plot the action samples from the sampler and scorer model
            self._add_actions(
                ax=fig.axes[0],
                action_data={"Sampled Actions": final_sample},
                action_step=action_step,
                colors=[plt.cm.get_cmap("tab10").colors[0]],
                zorder=3,
                scale=50
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

    def visualize_velocity_difference(
        self,
        scorer_velocity_model: nn.Module,
        sampler_global_cond: Tensor,
        scorer_global_cond: Tensor,
        velocity_eval_times: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None
    ):
        self.run_dir = self._update_run_dir()

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        sampler_global_cond = self._prepare_global_cond(sampler_global_cond)
        scorer_global_cond = self._prepare_global_cond(scorer_global_cond)

        # Initialize ODE solver
        sampler_ode_solver = ODESolver(self.velocity_model)
        scorer_ode_solver = ODESolver(scorer_velocity_model)

        # Sample noise from prior
        noise_sample = torch.randn(
            size=(self.num_paths, self.flow_matching_cfg.horizon, self.flow_matching_cfg.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator
        )
        
        # Create time grid
        if velocity_eval_times is None:
            velocity_eval_times = torch.linspace(0.0, 0.9, steps=10, device=device)           

        if self.flow_matching_cfg.ode_solver_method in FIXED_STEP_SOLVERS:
            sampling_time_grid = sampler_ode_solver.make_sampling_time_grid(
                step_size=self.flow_matching_cfg.ode_step_size,
                extra_times=velocity_eval_times,
                device=device,
            )
        elif self.flow_matching_cfg.ode_solver_method in ADAPTIVE_SOLVERS:
            sampling_time_grid, _ = torch.sort(torch.unique(velocity_eval_times)).to(device)
            # Append 1.0 if not already there
            if sampling_time_grid[-1].item() != 1.0:
                sampling_time_grid = torch.cat(
                    [sampling_time_grid, torch.tensor([1.0], device=device)]
                )

        # Sample paths from the ODE
        sampler_ode_states = sampler_ode_solver.sample(
            x_0=noise_sample,
            global_cond=sampler_global_cond.repeat(self.num_paths, 1),
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=sampling_time_grid,
            return_intermediate_states=True,
        ) # Shape: (timesteps, num_paths, horizon, action_dim)

        # Extract the final sampled action sequence
        final_sample = sampler_ode_states[-1] # Shape: (num_paths, horizon, action_dim)

        # Select the ODE states that correspond to the velocity evaluation times
        eval_ode_states, velocity_eval_times = sampler_ode_solver.select_ode_states(
            time_grid=sampling_time_grid,
            ode_states=sampler_ode_states,
            requested_times=velocity_eval_times,
        )
        paths = eval_ode_states.transpose(0, 1) # Shape: (num_paths, timesteps, horizon, action_dim)
        
        # Sample actions from the scorer model
        scorer_actions = scorer_ode_solver.sample(
            x_0=noise_sample,
            global_cond=scorer_global_cond.repeat(self.num_paths, 1),
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            step_size=self.flow_matching_cfg.ode_step_size,
        )

        # Compute velocity at each position for sampler and scorer
        sampler_vector_field = self._compute_vector_field(
            velocity_model=self.velocity_model,
            global_cond=sampler_global_cond,
            paths=paths,
            eval_times=velocity_eval_times,
        )
        scorer_vector_field = self._compute_vector_field(
            velocity_model=scorer_velocity_model,
            global_cond=scorer_global_cond,
            paths=paths,
            eval_times=velocity_eval_times,
        )
        
        # Select the specific action step and dimensions
        path_positions = paths[..., self.action_dims]
        sampler_velocity_vectors = sampler_vector_field[..., self.action_dims]
        scorer_velocity_vectors = scorer_vector_field[..., self.action_dims]

        # Compute global axis limits to create plots of equal size
        if self.axis_limits is None:
            self._compute_axis_limits(path_positions=path_positions)

        # Create a separate figure for each action step
        for action_step in self.action_steps:
            path_positions_single_action = path_positions[:, :, action_step, :]
            sampler_velocity_vectors_single_action = sampler_velocity_vectors[:, :, action_step, :]
            scorer_velocity_vectors_single_action = scorer_velocity_vectors[:, :, action_step, :]
            vector_fields = {
                "Sampler Velocities": sampler_velocity_vectors_single_action,
                "Scorer Velocities": scorer_velocity_vectors_single_action
            }

            if len(self.action_dims) == 2:
                fig, _ = plt.subplots(figsize=(12, 10))
                is_3d = False
            else:
                fig, _ = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
                is_3d = True

            self._add_actions(
                ax=fig.axes[0],
                action_data={"Sampler ODE States": sampler_ode_states.flatten(0,1)},
                action_step=action_step,
                colors=["grey"],
                scale=5,
                zorder=1
            )

            if not is_3d:
                fig = self._plot_flows_2d(
                    fig=fig,
                    path_positions=path_positions_single_action,
                    vector_fields=vector_fields,
                    time_grid=velocity_eval_times,
                    num_paths=self.num_paths,
                    action_step=action_step,
                )
            else:
                fig = self._plot_flows_3d(
                    fig=fig,
                    path_positions=path_positions_single_action,
                    vector_fields=vector_fields,
                    time_grid=velocity_eval_times,
                    num_paths=self.num_paths,
                    action_step=action_step,
                )

            # Plot the action samples from the sampler and scorer model
            self._add_actions(
                ax=fig.axes[0],
                action_data={"Sampler Actions": final_sample},
                action_step=action_step,
                colors=[plt.cm.get_cmap("tab10").colors[0]],
                zorder=3,
                scale=50
            )
            self._add_actions(
                ax=fig.axes[0],
                action_data={"Scorer Actions": scorer_actions},
                action_step=action_step,
                colors=[plt.cm.get_cmap("tab10").colors[1]],
                zorder=3,
                scale=50
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
        fig: Figure,
        path_positions: Tensor,
        vector_fields: Sequence[Tensor],
        time_grid: Tensor,
        num_paths: int,
        action_step: int,
    ) -> plt.Figure:
        """
        Draw a 3D quiver plot for the flow of a single action step.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        fig.canvas.manager.set_window_title("Visualization of Flows")
        ax = fig.axes[0]

        # Extract x-, y- and z-coordinates
        x, y, z = [path_positions[..., i].flatten().cpu() for i in range(3)]
        
        for field_idx, velocity_vectors in enumerate(vector_fields):
            # Extract velocities
            u, v, w = [velocity_vectors[..., i].flatten().cpu() for i in range(3)]

            # Scale original vectors by normalised time                                        # overall scale
            length_scale = 0.1
            u_scaled = u * time_norm
            v_scaled = v * time_norm
            w_scaled = w * time_norm

            time_norm = (time_grid / time_grid[-1]).repeat(num_paths).cpu()
            if len(vector_fields) > 1:
                # Pick a distinct base color for this field
                colors = plt.cm.get_cmap("tab10").colors[field_idx % 10]
            else:
                # Color arrows by time
                cmap = cm.get_cmap('viridis')
                colors = cmap(time_norm)

            # Create quiver plot
            quiv = ax.quiver(
                x, y, z,
                u_scaled, v_scaled, w_scaled, 
                length=length_scale,
                linewidth=1.5,
                arrow_length_ratio=0.25,
                normalize=False,
                color=colors
            )

        # Set consistent axis limits so the plots of all action steps have same size
        ax.set_xlim(*self.axis_limits[0])
        ax.set_ylim(*self.axis_limits[1])
        ax.set_zlim(*self.axis_limits[2])
        ax.set_aspect('equal')

        # Colorbar
        if len(vector_fields) > 1:
            cbar = fig.colorbar(quiv, ax=ax, shrink=0.95)
            cbar.ax.set_ylabel('Time', fontsize=12)
        # Title
        ax.set_title(
            f"Flow of Action Step {action_step+1} (Horizon: {self.flow_matching_cfg.horizon})",
            fontsize=16
        )

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
        fig: Figure,
        path_positions: Tensor,
        vector_fields: Dict[str, Tensor],
        time_grid: Tensor,
        num_paths: int,
        action_step: int,
    ) -> plt.Figure:
        """
        Draw a 2D quiver plot for the flow of a single action step.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        fig.canvas.manager.set_window_title("Visualization of Flows")
        ax = fig.axes[0]

        # Extract x- and y-coordinates
        x = path_positions[..., 0].flatten().cpu()
        y = path_positions[..., 1].flatten().cpu()

        if len(vector_fields) > 1:
            for field_idx, (label, velocity_vectors) in enumerate(vector_fields.items()):
                # Extract velocities
                u = velocity_vectors[..., 0].flatten().cpu()
                v = velocity_vectors[..., 1].flatten().cpu()
                
                # Pick a distinct base color for this field
                colour = plt.cm.get_cmap("tab10").colors[field_idx % 10]

                # Create quiver plot
                # Presentation: width=0.006
                quiv = ax.quiver(
                    x, y, u, v,
                    angles='xy', scale=len(time_grid),
                    scale_units='xy', width=0.004,
                    color=colour,
                    label=label
                )
        else:
            label, velocity_vectors = next(iter(vector_fields.items()))
            
            u = velocity_vectors[..., 0].flatten().cpu()
            v = velocity_vectors[..., 1].flatten().cpu()

            # Presentation: width=0.006
            quiv = ax.quiver(
                x, y, u, v, time_grid.repeat(num_paths).cpu(),
                angles='xy', scale=len(time_grid),
                scale_units='xy', width=0.004, cmap='viridis',
            )

            # Colorbar
            # Presentation
            # cbar.ax.set_ylabel('Time', fontsize=32, labelpad=12)
            # cbar.ax.tick_params(labelsize=28)
            cbar = fig.colorbar(quiv, ax=ax, shrink=0.95)
            cbar.ax.set_ylabel('Time', fontsize=12)

        # Set consistent axis limits so the plots of all action steps have same size
        ax.set_xlim(*self.axis_limits[0])
        ax.set_ylim(*self.axis_limits[1])
        ax.set_aspect('equal')

        # Title
        ax.set_title(
            f"Flow of Action Step {action_step+1} (Horizon: {self.flow_matching_cfg.horizon})",
            fontsize=16
        )

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
    

class VectorFieldVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting vector fields.
    """
    def __init__(
        self,
        cfg: VectorFieldVisConfig,
        flow_matching_cfg: FlowMatchingConfig,
        velocity_model: nn.Module,
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
        if len(cfg.action_dims) not in (2, 3):
            raise ValueError(
                "The vector-field visualisation supports 2D and 3D only, "
                f"(got action_dims = {cfg.action_dims}."
            )

        self.action_dims = cfg.action_dims
        self.action_dim_names = cfg.action_dim_names
        if cfg.action_steps is None:
            self.action_steps = list(range(self.flow_matching_cfg.horizon))
        else:
            self.action_steps = cfg.action_steps
        self.min_action = cfg.min_action
        self.max_action = cfg.max_action
        self.grid_size = cfg.grid_size
        # Default time_grid is list [0.05, 0.1, ..., 1.0]
        self.time_grid = list(np.linspace(0, 1, 21)) if cfg.time_grid is None else cfg.time_grid
        self.show = cfg.show
        self.vis_type = "vector_field"
    
    def visualize(
        self, global_cond: Tensor, generator: Optional[torch.Generator] = None, **kwargs
    ):
        """
        Visualize the 2D action vector field produced by a flow matching policy at a given time.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
                Shape [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.
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
        
        visualize_actions: bool = kwargs.get("visualize_actions", True)
        action_data = kwargs.get("actions", {})
        if visualize_actions and not action_data:
            # If no actions for visualization were passed in, sample some
            num_samples = 50 if len(self.action_dims) == 2 else 100
        else:
            # Only sample a single action sequence to create the vector field
            num_samples = 1

        # Sample noise vectors from prior
        noise_sample = torch.randn(
            size=(num_samples, self.flow_matching_cfg.horizon, self.flow_matching_cfg.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Sample action sequences
        action_samples = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond.repeat(num_samples, 1),
            step_size=self.flow_matching_cfg.ode_step_size,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
        )

        if visualize_actions:
            if not action_data:
                action_data["action_samples"] = action_samples[1:]
        
        if "base_action" not in action_data:
            action_data["base_action"] = action_samples[0].unsqueeze(0)

        # Build a 1-D lin-space once and reuse it for every axis we need
        axis_lin = np.linspace(self.min_action, self.max_action, self.grid_size)

        # Create the grids
        if len(self.action_dims) == 2:
            x_grid, y_grid = np.meshgrid(axis_lin, axis_lin, indexing="xy")
            x_dim, y_dim = self.action_dims
        else:
            x_grid, y_grid, z_grid = np.meshgrid(axis_lin, axis_lin, axis_lin, indexing="xy")
            x_dim, y_dim, z_dim = self.action_dims

        action_steps_idx = torch.randint(
            low=0,
            high=len(self.action_steps),
            size=(1,),
            generator=generator,
            device=device,
        ).item()
        action_step = self.action_steps[action_steps_idx]
        positions = action_data["base_action"].repeat(x_grid.size, 1, 1)
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
                    limits=(self.min_action, self.max_action),
                    action_step=action_step,
                    time=time,
                    max_velocity_norm=max_velocity_norm,
                    mean_uncertainty=kwargs.get("mean_uncertainty", None)
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
                    mean_uncertainty=kwargs.get("mean_uncertainty", None)
                )

            if visualize_actions:
                self._add_actions(
                    ax=fig.axes[0],
                    action_data=action_data,
                    action_step=action_step,
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
        max_velocity_norm: float,
        mean_uncertainty: Optional[float] = None,
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

        if mean_uncertainty:
            ax.text2D(
                0.02, 0.98,
                f"Mean Uncertainty: {mean_uncertainty:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8
                ),
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
        mean_uncertainty: Optional[float] = None,
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

        if mean_uncertainty:
            ax.text(
                0.02, 0.98,
                f"Mean uncertainty: {mean_uncertainty:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8
                ),
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