from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.common.flow_matching.ode_solver import ODESolver
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.visualizer.configuration_visualizer import ActionSeqVisConfig

from .visualizer import FlowMatchingVisualizer


# TODO: Add action sequence visualization to live stream and video of policy rollout
class ActionSeqVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting a batch of action sequences onto the current observation frame.
    """
    def __init__(
        self,
        config: ActionSeqVisConfig,
        model: BaseFlowMatchingAdapter,
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        output_root: Optional[Union[Path, str]],
        save: bool = True,
        create_gif: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            postprocessor: A pipeline that processes the raw policy actions into the format expected
                by the environment
            num_action_seq: Number of action sequences to plot.
        """
        super().__init__(
            model=model,
            save=save,
            output_root=output_root,
            create_gif=create_gif,
            verbose=verbose,
        )
        self.num_action_seq = config.num_action_seq
        self.postprocessor = postprocessor
        self.show = config.show
        self.index_runs = False
        self.vis_type = "action_seq"

    def visualize(
        self, observation: Dict[str, Tensor], generator: Optional[torch.Generator] = None, **kwargs
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

        dir_name = kwargs.get("dir_name")
        self.run_dir = self._update_run_dir(vis_type_dir_name=dir_name)

        # Initialize ODE solver
        ode_solver = ODESolver()

        # Build the velocity function conditioned on the current observation
        conditioning = self.model.prepare_conditioning(observation, self.num_action_seq)
        velocity_fn = self.model.make_velocity_fn(conditioning=conditioning)

        # Sample noise vectors from prior
        noise_sample = self.model.sample_prior(
            num_samples=self.num_action_seq,
            generator=generator
        )

        # Sample a batch of action sequences
        actions = ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=velocity_fn,
            method=self.model.ode_solver_config["solver_method"],
            step_size=self.model.ode_solver_config["step_size"],
            atol=self.model.ode_solver_config["atol"],
            rtol=self.model.ode_solver_config["rtol"],
        )

        # Convert the model-predicted actions from the network’s normalised space
        # back into world coordinates
        actions = self.postprocessor(actions)

        # Plotting of action sequences depends on environment
        if env.spec is None:
            env_namespace = env._namespace
        else:
            env_namespace = env.spec.namespace
        if env_namespace == "gym_aloha":
            fig = self._create_aloha_action_seq_image(env=env, actions=actions.cpu())
        elif env_namespace == "gym_pusht":
            frame = env.render()
            fig = self._create_pusht_action_seq_image(frame=frame, actions=actions.cpu())
        elif env_namespace == "gym_libero":
            fig = self._create_libero_action_seq_image(env=env, actions=actions.cpu())
        else:
            raise ValueError(
                f"ActionSeqVisualizer does not support environment with namespace '{env_namespace}'."
            )

        if self.show:
            plt.show(block=True)
        if self.save:
            self._save_figure(fig)

        plt.close(fig)

    def _project_world_point_to_pixels(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
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
            mj_model._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        cam_pos = mj_data.cam_xpos[cam_id]
        cam_mat = mj_data.cam_xmat[cam_id].reshape(3, 3)

        # Convert point from world to camera coordinates
        p_cam = cam_mat.T @ (point_world - cam_pos)

        # Everything in front of the camera has negative z in MuJoCo
        if p_cam[2] >= 0:
            raise ValueError("Point is behind the camera")

        # Get the camera intrinsics
        fovy_deg = mj_model.cam_fovy[cam_id]
        fovy_rad = np.deg2rad(fovy_deg)
        f = 0.5 * img_height / np.tan(0.5 * fovy_rad)

        # Get the image centre
        cx, cy = img_width * 0.5, img_height * 0.5

        # Perspective projection
        u = (-p_cam[0] / -p_cam[2]) * f + cx
        v = (-p_cam[1] / -p_cam[2]) * f + cy

        return u, v

    def _draw_waypoints(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        waypoints: List[np.ndarray],
        ax: Axes
    ):
        # Prepare a colormap over the trajectory length
        norm = plt.Normalize(0, self.model.horizon - 1)
        cmap = plt.get_cmap("turbo")

        pixel_points: List[Tuple[float, float]] = []
        for point_3d in waypoints:
            try:
                u_px, v_px = self._project_world_point_to_pixels(
                    mj_model,
                    mj_data,
                    point_3d,
                    camera_name="agentview",
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
        line_collection.set_array(np.arange(self.model.horizon - 1))
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

            for action_step in range(self.model.horizon):
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
        for waypoints_left, waypoints_right in zip(all_waypoints_left, all_waypoints_right, strict=False):
            mj_model, mj_data = env.unwrapped._env.physics.model, env.unwrapped._env.physics.data
            self._draw_waypoints(mj_model=mj_model, mj_data=mj_data, waypoints=waypoints_left, ax=ax)
            self._draw_waypoints(mj_model=mj_model, mj_data=mj_data, waypoints=waypoints_right, ax=ax)

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

        libero_env = env._env # OffScreenRenderEnv (LIBERO wrapper)
        base_env = libero_env.env # RoboSuite BaseEnv

        # Save the initial MuJoCo state (qpos + qvel) so we can restore later
        initial_state = libero_env.get_sim_state()

        # Render the current camera image (RGB), using the "top" camera
        frame = env.render()

        # Tell the LIBERO/robosuite wrapper to ignore the “episode done” flag,
        # so we can keep stepping through from the same sim state without triggering a reset.
        base_env.ignore_done = True

        # Prepare to store 3D waypoints for each action sequence
        all_waypoints: List[List[np.ndarray]] = []

        for seq_idx in range(actions.shape[0]):
            # Restore the saved state before simulating this sequence
            libero_env.set_state(initial_state)
            base_env.sim.forward()
            libero_env.check_success()
            base_env._post_process()
            base_env._update_observables(force=True)

            seq_waypoints: List[np.ndarray] = []

            action_seq = actions[seq_idx].cpu().numpy()
            for action_step in range(self.model.horizon):
                obs, _, _, _ = libero_env.step(action_seq[action_step])
                end_effector_pos = obs["robot0_eef_pos"]
                seq_waypoints.append(end_effector_pos)

            all_waypoints.append(seq_waypoints)

        libero_env.set_state(initial_state)
        base_env.sim.forward()
        libero_env.check_success()
        base_env._post_process()
        base_env._update_observables(force=True)

        # Create a Matplotlib figure and axis with the same extents as the image
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax.imshow(frame)
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)  # invert y-axis so that v increases downward
        ax.set_aspect("equal")
        ax.axis("off")

        # For each sequence, project 3D waypoints to pixel coords and draw
        for waypoints in all_waypoints:
            mj_model, mj_data = env._env.sim.model, env._env.sim.data
            self._draw_waypoints(mj_model=mj_model, mj_data=mj_data, waypoints=waypoints, ax=ax)

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
        norm = plt.Normalize(0, self.model.horizon - 1)
        cmap = plt.get_cmap("turbo")

        # Draw each action sequence as a colourful line strip
        for action_seq in actions:
            segments = [
                [tuple(action_seq[i]), tuple(action_seq[i+1])]
                for i in range(self.model.horizon - 1)
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
            lc.set_array(np.arange(self.model.horizon - 1))
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

    def get_figure_filename(self, **kwargs) -> str:
        """Get the figure filename of the current rollout step."""
        action_seq_idx = 1
        while (self.run_dir / f"action_seqs_{action_seq_idx:04d}.png").exists():
            action_seq_idx += 1
        filename = f"action_seqs_{action_seq_idx:04d}"

        return filename
