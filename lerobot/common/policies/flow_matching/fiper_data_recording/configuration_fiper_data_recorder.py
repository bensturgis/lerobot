from dataclasses import dataclass, field
from pathlib import Path


def process_ode_eval_times(
    ode_eval_times: list[float],
) -> list[float]:
    """
    Clean and normalize ODE evaluation times.

    - Removes endpoints (0.0 and 1.0) and any values outside (0, 1).
    - Sorts the list in ascending order and removes duplicates.
    """
    # Remove endpoints 0 and 1 as well as points outside the range of (0,1)
    processed_ode_eval_times: list[float] = []
    for t in ode_eval_times:
        if round(t, 3) <= 0.0 or round(t, 3) >= 1.0:
            continue
        processed_ode_eval_times.append(round(t, 3))

    # Sort and remove duplicates
    processed_ode_eval_times = sorted(set(processed_ode_eval_times))

    return processed_ode_eval_times


@dataclass
class FiperDataRecorderConfig:
    # Number of action sequences sampled per observation for uncertainty estimation
    num_uncertainty_sequences: int = 256

    # Times at which to record ODE states and velocities during the flow integration
    ode_eval_times: list[float] = field(
        default_factory=lambda: [round(i * 0.05, 2) for i in range(20)] + [1.0]
    )
    
    # Parameters for the ensemble model
    ensemble_model_path: str | Path | None = None

    # Parameters for the Laplace approximation calibration dataloader
    laplace_scope: str = "both"
    calib_fraction: float = 1.0
    batch_size: int = 1

    def __post_init__(self) -> None:
        self.ode_eval_times = process_ode_eval_times(self.ode_eval_times)