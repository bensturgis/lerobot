from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

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

def add_action_overlays(
    ax: Axes,
    action_data: Dict[str, Tensor],
    action_step: int,
    action_dims: Sequence[int],
    colors: Optional[Iterable[str]] = None,
    zorder: int = 3,
    scale: float = 10.0,
) -> Axes:
    """
    Overlay action samples on flow matching visualizations.
    """
    if colors is None:
        colors = ["red", "orange", "green", "purple", "magenta", "brown"]
    
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