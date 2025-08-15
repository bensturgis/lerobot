import matplotlib.pyplot as plt
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from torch import nn, Tensor
from typing import Optional

from .utils import make_gif, make_run_dir, next_available_name
from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig


class FlowMatchingVisualizer(ABC):
    """
    Abstract base class for flow matching visualizer.
    """
    vis_type: str = "base"
    index_runs: bool = True
    gif_name_base: Optional[str] = None

    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        velocity_model: nn.Module,
        save: bool,
        output_root: Optional[Path],
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
        self.output_root = Path(output_root) if output_root else Path("outputs/visualizations")
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

    @abstractmethod
    def get_figure_filename(self, **kwargs) -> str:
        """Get filename for a figure based on the visualization type."""
        pass

    def log(self, msg: str) -> None:
        """Print a message only if verbose is set to True."""
        if self.verbose:
            print(msg)


    def _update_run_dir(
        self, vis_type_dir_name: Optional[str] = None
    ) -> Path:
        """
        Create a new, empty folder and return its path.
        """       
        if vis_type_dir_name is None:
            vis_type_dir = self.output_root / self.vis_type
        else:
            vis_type_dir = self.output_root / vis_type_dir_name
        self.run_dir = make_run_dir(base_dir=vis_type_dir, indexed=self.index_runs)

        return self.run_dir
    
    def _save_figure(
        self,
        fig: plt.Figure,
        **kwargs
    ):
        """
        Save the given figure to a visualization directory based on the visualization type.
        """
        if not self.save:
            return
        filename = self.get_figure_filename(**kwargs)
        filepath = self.run_dir / filename
        if filepath.exists():
            self.log(f"Warning: File {filepath} already exists. Skipping save.")
        else:
            # Presentation
            # fig.savefig(filepath, dpi=600)
            fig.savefig(filepath, dpi=300)
            self.log(f"Saved figure to {filepath}.")

    def _create_gif(self, duration: float = 0.2):
        """
        Create an animated GIF from a sequence of saved flow
        or vector field plot images.
        """
        if not (self.save and self.create_gif and self.gif_name_base):
            return

        # Get the path where the GIF will be saved
        vis_type_dir = self.output_root / self.vis_type

        # Get the path to the GIF file
        gif_path = next_available_name(vis_type_dir, self.gif_name_base, ext="gif")

        # Make the GIF with the frames from the current run dir
        make_gif(frames_dir=self.run_dir, out_path=gif_path, duration=duration)

        self.log(f"Saved GIF to {gif_path}")