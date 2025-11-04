from dataclasses import dataclass, field


def process_ode_eval_times(
    ode_eval_times: list[float],
) -> list[float]:
    """
    Clean and normalize ODE evaluation times.

    - Removes any values outside (0, 1).
    - Sorts the list in ascending order and removes duplicates.
    """
    # Remove points outside the range of (0,1)
    processed_ode_eval_times: list[float] = []
    for t in ode_eval_times:
        if round(t, 3) < 0.0 or round(t, 3) > 1.0:
            continue
        processed_ode_eval_times.append(round(t, 3))

    # Sort and remove duplicates
    processed_ode_eval_times = sorted(set(processed_ode_eval_times))

    return processed_ode_eval_times

@dataclass
class LikelihoodODESolverConfig:
    """Configuration for the ODE solver used in likelihood estimation."""
    method: str = "euler"
    atol: float | None = None
    rtol: float | None = None
    exact_divergence: bool = False

@dataclass
class LaplaceConfig:
    """Configuration parameters for the Laplace approximation."""
    scopes: list[str] | None = None
    calib_fraction: float = 1.0
    batch_size: int = 1
    num_samples: int = 5

@dataclass
class FiperDataRecorderConfig:
    """
    Configuration for recording rollout data to evaluate failure prediction capabilities of
    uncertainty estimation methods using framework FIPER:
    https://github.com/ralfroemer99/fiper.
    """
    # Number of action sequences sampled per observation for uncertainty estimation
    num_uncertainty_sequences: int = 256

    # Which environment domains to evaluate. Allowed values: "id" (in-distribution), "ood" (out-of-distribution).
    domains: list[str] = field(default_factory=lambda: ["id", "ood"])

    # Scoring metrics to be recorded, choose from "mode_distance", "inter_vel_diff" and "likelihood"
    scoring_metrics: tuple = ("mode_distance", "inter_vel_diff", "likelihood")

    # Parameters for the ensemble model
    ensemble_model_paths: list[str] | None = None

    # Parameters for the Laplace approximation
    laplace_config: LaplaceConfig = field(default_factory=LaplaceConfig)

    # Times at which to record ODE states and velocities during the flow integration
    ode_eval_times: list[float] = field(
        default_factory=lambda: [round(i * 0.05, 2) for i in range(20)] + [1.0]
    )

    # Times at which to evaluate the velocity of the sampled action sequence under a scorer model
    terminal_vel_eval_times: list[float] = field(
        default_factory=lambda: [round(i * 0.1, 1) for i in range(10)]
    )

    # Configuration for the ODE solver used in likelihood estimation
    likelihood_ode_solver_cfg: LikelihoodODESolverConfig = field(
        default_factory=LikelihoodODESolverConfig
    )

    def __post_init__(self) -> None:
        self.ode_eval_times = process_ode_eval_times(self.ode_eval_times)
