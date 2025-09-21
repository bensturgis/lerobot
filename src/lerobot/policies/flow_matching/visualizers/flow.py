from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.modelling_flow_matching import FlowMatchingConditionalUnet1d
from lerobot.common.policies.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
    make_sampling_time_grid,
    select_ode_states,
)
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.configs.default import FlowVisConfig

from .base import FlowMatchingVisualizer
from .utils import add_actions, compute_axis_limits


class FlowVisualizer(FlowMatchingVisualizer):
    """
    Visualizer for plotting flow trajectories through the learned velocity field.
    """
    def __init__(
        self,
        cfg: FlowVisConfig,
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
        self.gif_name_base = "flow_animation"
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

    def _make_sampling_grid(
        self, 
        ode_solver: ODESolver,
        vel_eval_times: Optional[Tensor],
        device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """
        Create the grid containing the times at which the velocities will be evaluated and plotted
        as well as the sampling time grid for solving the ODE.
        """
        # Default eval times
        if vel_eval_times is None:
            vel_eval_times = torch.arange(0.0, 0.9 + 1e-8, 0.1, device=device)
        else:
            vel_eval_times = vel_eval_times.to(device)

        # Always unique + sorted for safety
        vel_eval_times = torch.unique(vel_eval_times)
        vel_eval_times, _ = torch.sort(vel_eval_times)

        method = self.flow_matching_cfg.ode_solver_method
        if method in FIXED_STEP_SOLVERS:
            sampling_time_grid = make_sampling_time_grid(
                step_size=self.flow_matching_cfg.ode_step_size,
                extra_times=vel_eval_times,
                device=device,
            )
        elif method in ADAPTIVE_SOLVERS:
            sampling_time_grid = vel_eval_times
            if sampling_time_grid[-1].item() != 1.0:
                sampling_time_grid = torch.cat([sampling_time_grid, sampling_time_grid.new_tensor([1.0])])
        else:
            raise ValueError(f"Unknown ODE method: {method}")

        return vel_eval_times, sampling_time_grid

    def _compute_vector_field(
        self, velocity_model: FlowMatchingConditionalUnet1d, global_cond: Tensor, paths: Tensor, eval_times: Tensor, 
    ) -> Tensor:
        """
        Compute the model's velocity vectors along each trajectory.
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

    def _make_figure(self) -> Tuple[Figure, bool]:
        """Create 2D or 3D figure based on action_dims."""
        if len(self.action_dims) == 2:
            fig, _ = plt.subplots(figsize=(12, 10))
            return fig, False
        fig, _ = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
        return fig, True

    def _render_action_steps(
        self,
        path_positions: Tensor,
        vector_fields: Dict[str, Tensor],
        eval_times: Tensor,
        ode_states: Tensor,
        final_action_overlays: List[Dict[str, Any]],                # list of dicts: {name, tensor, color, scale, zorder}
    ):
        """Render the flow plots including overlaid actions."""
        # Compute global axis limits to create plots of equal size
        if self.axis_limits is None:
            self.axis_limits = compute_axis_limits(ode_states=path_positions, action_dims=self.action_dims)

        # Create a separate figure for each action step
        for action_step in self.action_steps:
            single_action_path_positions = path_positions[:, :, action_step, :]
            single_action_vector_field = {label: vel[:, :, action_step, :] for label, vel in vector_fields.items()}

            fig, is_3d = self._make_figure()

            # Overlay ODE states as background
            add_actions(
                ax=fig.axes[0],
                action_data={"Sampler ODE States": ode_states.flatten(0, 1)},
                action_step=action_step,
                action_dims=self.action_dims,
                colors=["grey"],
                scale=5,
                zorder=1
            )

            # Vector field plot
            fig = self._plot_flows(
                fig=fig,
                path_positions=single_action_path_positions,
                vector_fields=single_action_vector_field,
                time_grid=eval_times,
                action_step=action_step,
            )

            # Overlay the action samples from the sampler and scorer model
            for overlay in final_action_overlays:
                add_actions(
                    ax=fig.axes[0],
                    action_data={overlay["name"]: overlay["tensor"]},
                    action_step=action_step,
                    action_dims=self.action_dims,
                    colors=[overlay["color"]],
                    zorder=overlay.get("zorder", 3),
                    scale=overlay.get("scale", 30),
                )

            # Draw the legend
            fig.axes[0].legend()

            # Show plot if requested
            if self.show:
                plt.show(block=True)

            # Save plot if requested
            if self.save:
                self._save_figure(fig, action_step=action_step)

            plt.close(fig)

        if self.create_gif:
            self._create_gif()

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
        
        # Create time grid for solving the ODE
        vel_eval_times, sampling_time_grid = self._make_sampling_grid(
            ode_solver=ode_solver,
            vel_eval_times=None,
            device=device,
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
        eval_ode_states, vel_eval_times = select_ode_states(
            time_grid=sampling_time_grid,
            ode_states=ode_states,
            requested_times=vel_eval_times,
        )
        paths = eval_ode_states.transpose(0, 1) # Shape: (num_paths, timesteps, horizon, action_dim)
        
        # Compute velocity at each position
        vector_field = self._compute_vector_field(
            velocity_model=self.velocity_model,
            global_cond=global_cond,
            paths=paths,
            eval_times=vel_eval_times,
        )
        vector_field_dict = {
            "Velocities": vector_field
        }
        
        # Select the specific action step and dimensions
        path_positions = paths[..., self.action_dims]
        
        self._render_action_steps(
            path_positions=path_positions,
            vector_fields=vector_field_dict,
            eval_times=vel_eval_times,
            ode_states=ode_states,
            final_action_overlays=[{
                "name": "Sampled Actions",
                "tensor": final_sample,
                "color": "red",
                "scale": 30,
                "zorder": 3,
            }],
        )

    def visualize_velocity_difference(
        self,
        scorer_velocity_model: FlowMatchingConditionalUnet1d,
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
        
        # Create time grid for solving the ODE
        velocity_eval_times, sampling_time_grid = self._make_sampling_grid(
            ode_solver=sampler_ode_solver,
            vel_eval_times=velocity_eval_times,
            device=device,
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
        eval_ode_states, velocity_eval_times = select_ode_states(
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
        vector_fields = {
            "Sampler Velocities": sampler_vector_field,
            "Scorer Velocities": scorer_vector_field
        }
        
        # Select the specific action step and dimensions
        path_positions = paths[..., self.action_dims]

        # Brighten colors for action overlays
        def _brighten(rgb, f): return np.clip(np.array(rgb) * f, 0, 1)
        c0 = _brighten(plt.cm.get_cmap("tab10").colors[0], 1.6)
        c1 = _brighten(plt.cm.get_cmap("tab10").colors[1], 1.4)

        self._render_action_steps(
            path_positions=path_positions,
            vector_fields=vector_fields,
            eval_times=velocity_eval_times,
            ode_states=sampler_ode_states,
            final_action_overlays=[
                {"name": "Sampler Actions", "tensor": final_sample, "color": c0, "scale": 50, "zorder": 3},
                {"name": "Scorer Actions",  "tensor": scorer_actions, "color": c1, "scale": 50, "zorder": 3},
            ],
        )

    def _plot_flows(
        self,
        fig: Figure,
        path_positions: torch.Tensor,      # (num_paths, T, horizon, d) with d=2 or 3
        vector_fields: Dict[str, torch.Tensor],
        time_grid: torch.Tensor,
        action_step: int,
    ) -> plt.Figure:
        """
        Draw a 2D or 3D quiver plot for the flow of a single action step.
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        fig.canvas.manager.set_window_title("Visualization of Flows")
        ax = fig.axes[0]

        # Extract x-, y- and z-coordinates
        dim = path_positions.shape[-1]
        coords = [path_positions[..., i].flatten().cpu() for i in range(dim)]
        x, y = coords[0], coords[1]
        z = coords[2] if dim == 3 else None

        # Scale velocity vectors
        time_diff = torch.diff(time_grid, append=time_grid.new_tensor([1.0]))
        vector_fields = {label: vel * time_diff.view(1, -1, 1) for label, vel in vector_fields.items()}

        quiv = None
        if len(vector_fields) > 1:
            for field_idx, (label, velocity_vectors) in enumerate(vector_fields.items()):
                vel_components = [velocity_vectors[..., i].flatten().cpu() for i in range(dim)]
                colour = plt.cm.get_cmap("tab10").colors[field_idx % 10]

                if dim == 2:
                    quiv = ax.quiver(
                        x, y, vel_components[0], vel_components[1],
                        angles='xy', scale_units='xy', width=0.004,
                        color=colour, label=label
                    )
                else:
                    quiv = ax.quiver(
                        x, y, z,
                        vel_components[0], vel_components[1], vel_components[2],
                        linewidth=1.5, arrow_length_ratio=0.25,
                        normalize=False, color=colour, label=label
                    )
        else:
            # Single field: color by time
            (label, velocity_vectors), = vector_fields.items()
            vel_components = [velocity_vectors[..., i].flatten().cpu() for i in range(dim)]
            time_norm = (time_grid / time_grid[-1]).repeat(self.num_paths).cpu()
            if dim == 2:
                # Presentation: width=0.006
                quiv = ax.quiver(
                    x, y, vel_components[0], vel_components[1],
                    time_norm, angles='xy', scale_units='xy', width=0.004,
                    cmap='viridis', scale=1.0
                )
            else:
                colors = cm.get_cmap('viridis')(time_norm)
                quiv = ax.quiver(
                    x, y, z,
                    vel_components[0], vel_components[1], vel_components[2],
                    linewidth=1.5, arrow_length_ratio=0.25,
                    normalize=False, color=colors, label=None
                )
            # cbar.ax.set_ylabel('Time', fontsize=32, labelpad=12)
            # cbar.ax.tick_params(labelsize=28)
            cbar = fig.colorbar(quiv, ax=ax, shrink=0.95)
            cbar.ax.set_ylabel('Time', fontsize=12)

        # Axis labels
        dims = self.action_dims[:dim]
        if self.action_dim_names:
            labels = [self.action_dim_names[d] for d in dims]
        else:
            labels = [f"Action dimension {d}" for d in dims]
        ax.set_xlabel(labels[0], fontsize=14)
        ax.set_ylabel(labels[1], fontsize=14)
        if dim == 3:
            ax.set_zlabel(labels[2], fontsize=14, labelpad=8)

        ax.set_xlim(*self.axis_limits[0])
        ax.set_ylim(*self.axis_limits[1])
        if dim == 3:
            ax.set_zlim(*self.axis_limits[2])
        ax.set_aspect('equal')

        # Title
        ax.set_title(
            f"Flows of Action Step {action_step+1} (Horizon: {self.flow_matching_cfg.horizon})",
            fontsize=16
        )
        # Presentation
        # ax.tick_params(
        #     axis='both',
        #     which='both',
        #     labelbottom=False,
        #     labelleft=False
        # )
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        plt.tight_layout()

        if was_interactive:
            plt.ion()

        return fig
    
    def get_figure_filename(self, **kwargs) -> str:
        """Get the figure filename for the current action step."""
        if "action_step" not in kwargs:
            raise ValueError(
                "`action_step` must be provided to get filename of flows figure."
            )
        action_step = kwargs["action_step"]

        return f"flows_action_{action_step+1:02d}.png"