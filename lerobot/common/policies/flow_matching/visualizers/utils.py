from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import imageio
from matplotlib.axes import Axes
from torch import Tensor


def make_run_dir(base_dir: Path, indexed: bool):
    """
    Create new folder for the current visualization iteration during a rollout.
    """
    base_dir.mkdir(parents=True, exist_ok=True)            
    if indexed:
        run_idx = 1
        while (base_dir / f"{run_idx:03d}").exists():
            run_idx += 1
        run_dir = base_dir / f"{run_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    else:
        return base_dir
    
def make_gif(
    frames_dir: Path, out_path: Path, duration: float = 0.2
) -> Path:
    """
    Build an animated GIF from frame images in frames_dir.
    """
    # Build the list of filenames in the right order
    filepaths = sorted([fp for fp in frames_dir.iterdir() if fp.suffix.lower() == ".png"])

    # Read each frame
    frames = [imageio.imread(str(fp)) for fp in filepaths]

    # Save the GIF
    imageio.mimsave(str(out_path), frames, duration=duration)

    
def next_available_name(base_dir: Path, file_name_base: str, ext: str) -> str:
    """Get the next available file name for some base name in the base_dir."""
    run_idx = 1
    while (base_dir / f"{file_name_base}_{run_idx:03d}.{ext}").exists():
        run_idx += 1
    return base_dir / f"{file_name_base}_{run_idx:03d}.gif"

def add_actions(
    ax: Axes,
    action_data: Dict[str, Tensor],
    action_step: int,
    action_dims: int,
    colors: Iterable[str] = [],
    zorder: int = 3,
    scale: float = 10.0,
    marker: str = "o",
    step_label: Optional[str] = None,
    text_kwargs: Optional[Dict[str, Any]] = None,
) -> Axes:
    """
    Overlay action samples on flow matching visualizations.
    """
    if len(colors) == 0:
        colors = ["red", "orange", "green", "purple", "magenta", "brown"]
    
    if text_kwargs is None:
        text_kwargs = {}

    # Figure out which dims to plot
    if len(action_dims) == 2:
        x_dim, y_dim = action_dims
        is_3d = False
    else:
        x_dim, y_dim, z_dim = action_dims
        is_3d = True

    # Plot each action sequence
    for idx, (name, actions) in enumerate(action_data.items()):
        color = colors[idx % len(colors)]
        # Actions shape: (num_samples, horizon, action_dim)
        x_positions = actions[:, action_step, x_dim].cpu().numpy()
        y_positions = actions[:, action_step, y_dim].cpu().numpy()
        label = name.replace("_", " ")

        existing_labels = {lab for lab in ax.get_legend_handles_labels()[1] if not lab.startswith("_")}
        label = label if label not in existing_labels else "_nolegend_"
        if is_3d:
            z_positions = actions[:, action_step, z_dim].cpu().numpy()
            # Presentation s=30
            ax.scatter(
                x_positions, y_positions, z_positions, label=label,
                color=color, s=scale, zorder=zorder, marker=marker,
            )
        else:
            # Presentation s=30
            ax.scatter(
                x_positions, y_positions, label=label,
                color=color, s=scale, zorder=zorder,
                marker=marker,
            )

        # Optional per-point labels
        if step_label is not None:
            num_points = len(x_positions)
            for i in range(num_points):
                if is_3d:
                    ax.text(
                        x_positions[i], y_positions[i], z_positions[i],
                        step_label, zorder=zorder + 1)
                else:
                    ax.annotate(
                        step_label, (x_positions[i], y_positions[i]),
                        xytext=text_kwargs.get("xytext", (0.2, 0.2)),
                        textcoords="offset points", zorder=zorder + 1,
                        fontsize=11
                    )

    # Presentation: fontsize=28
    return ax.get_figure()

def compute_axis_limits(ode_states: Tensor, action_dims: Sequence[int]) -> List[int]:
    """Compute global axis limits to create plots of equal size."""
    axis_limits = []
    for i in range(len(action_dims)):
        coords = ode_states[..., i]           # (num_samples, t, steps)
        coord_min, coord_max = coords.min().cpu(), coords.max().cpu()
        margin = 0.05 * (coord_max - coord_min)
        axis_limits.append((coord_min - margin, coord_max + margin))

    return axis_limits