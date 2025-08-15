import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch

from matplotlib.figure import Figure
from pathlib import Path
from torch import nn, Tensor
from typing import Dict, Optional, Sequence, Union

from .base import FlowMatchingVisualizer
from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
)
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.configs.default import FlowVisConfig


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

    def _compute_vector_field(
        self, velocity_model: nn.Module, global_cond: Tensor, paths: Tensor, eval_times: Tensor, 
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

    def get_figure_filename(self, **kwargs) -> str:
        """Get the figure filename of the current step in the action sequence."""
        if "action_step" not in kwargs:
            raise ValueError(
                "`action_step` must be provided to get filename of flows figure."
            )
        
        action_step = kwargs["action_step"]
        return f"flows_action_{action_step+1:02d}.png"