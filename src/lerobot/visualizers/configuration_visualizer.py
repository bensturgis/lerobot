from dataclasses import dataclass
from typing import Optional


@dataclass
class FlowVisConfig:
    # Two or three indices indicating which action dimensions to visualize
    action_dims: list[int] = (0,1)

    # Names of the action dimensions to visualize
    action_dim_names: Optional[list[str]] = None

    # Time‐step indices (horizon steps) at which to generate visualizations
    action_steps: Optional[list[int]] = None

    # Whether to display plots live
    show: bool = False

    # Custom axis limits for each plotted dimension as a list of (min, max) tuples
    axis_limits: Optional[list[tuple[float, float]]] = None

    # Number of trajectory samples to draw when visualizing multiple action sequences
    num_paths: int = 50

@dataclass
class VectorFieldVisConfig:
    # Two or three indices indicating which action dimensions to visualize
    action_dims: list[int] = (0,1)

    # Names of the action dimensions to visualize
    action_dim_names: Optional[list[str]] = None

    # Time‐step indices (horizon steps) at which to generate visualizations
    action_steps: Optional[list[int]] = None

    # Whether to display plots live
    show: bool = False

    # Minimum value of the action space (default −1.0 after normalization)
    min_action: float = -1.0

    # Maximum value of the action space (default +1.0 after normalization)
    max_action: float = 1.0

    # Number of grid points per axis when sampling the action space for the quiver plot
    grid_size: int = 50

    # List of time values (between 0 and 1) at which to compute and draw the vector field
    time_grid: Optional[list[float]] = None

@dataclass
class ActionSeqVisConfig:
    # Whether to display plots live
    show: bool = False

    # How many action sequences to visualize
    num_action_seq: int = 30

@dataclass
class NoiseToActionVisConfig:
    # Whether to display plots live
    show: bool = False

    # How many noise samples to action sequences to visualize
    num_samples: int = 3

    # Two indices indicating which action dimensions to visualize
    action_dims: list[int] = (0,1)

    # Names of the action dimensions to visualize
    action_dim_names: Optional[list[str]] = None

    # Custom axis limits for each plotted dimension as a list of (min, max) tuples
    axis_limits: Optional[list[tuple[float, float]]] = None

    # Time points along the ODE integration where noisy actions are visualized
    ode_eval_times: list | tuple  = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
