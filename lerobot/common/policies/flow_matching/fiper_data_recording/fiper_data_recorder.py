import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal

from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

from ..configuration_flow_matching import FlowMatchingConfig
from ..modelling_flow_matching import FlowMatchingModel
from ..ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
    make_lik_estimation_time_grid,
    make_sampling_time_grid,
)
from ..uncertainty.laplace_utils import draw_laplace_flow_matching_model
from ..uncertainty.scorer_artifacts import ScorerArtifacts
from .configuration_fiper_data_recorder import FiperDataRecorderConfig


class FiperDataRecorder:
    """
    Records data for evaluate failure prediction capabilities of uncertainty estimation
    methods using framework FIPER:
    https://github.com/ralfroemer99/fiper.
    """
    def __init__(
        self,
        config: FiperDataRecorderConfig,
        flow_matching_config: FlowMatchingConfig,
        flow_matching_model: FlowMatchingModel,
        scorer_artifacts: ScorerArtifacts,
    ):
        self.config = config
        self.flow_matching_config = flow_matching_config
        self.flow_matching_model = flow_matching_model
        self.ode_solver = ODESolver(velocity_model=self.flow_matching_model.unet)

        self.horizon = self.flow_matching_config.horizon
        self.action_dim = self.flow_matching_config.action_feature.shape[0]
        self.device = get_device_from_parameters(self.flow_matching_model)
        self.dtype = get_dtype_from_parameters(self.flow_matching_model)

        # Extract scorer artifacts
        self.ensemble_model = scorer_artifacts.ensemble_model
        self.laplace_posterior = scorer_artifacts.laplace_posterior
        if self.ensemble_model is None:
            raise ValueError("Ensemble model is required for FIPER data recording.")
        elif self.laplace_posterior is None:
            raise ValueError("Laplace posterior is required for FIPER data recording.")
        
        # Build time grid for sampling according to ODE solver method and scoring metric
        if self.flow_matching_config.ode_solver_method in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = make_sampling_time_grid(
                step_size=self.flow_matching_config.ode_step_size,
                extra_times=self.config.ode_eval_times,
                device=self.device
            )
        elif self.flow_matching_config.ode_solver_method in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = torch.tensor(
                [0.0, *([] if self.config.ode_eval_times is None else self.config.ode_eval_times), 1.0],
                device=self.device, dtype=self.dtype
            )
        else:
            raise ValueError(f"Unknown ODE solver method: {self.flow_matching_config.ode_solver_method}.")
        
        # Noise distribution is an isotropic Gaussian
        self.gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(self.horizon, self.action_dim, device=self.device, dtype=self.dtype),
                scale = torch.ones(self.horizon, self.action_dim, device=self.device, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob
        
        # Build time grid for likelihood estimation based on ODE solver method
        self.lik_estimation_time_grid = make_lik_estimation_time_grid(
            ode_solver_method=self.config.likelihood_ode_solver_cfg.method,
            device=self.device,    
        )
        
        self.rollout_data: List[Dict[str, Any]] = []

    def compute_log_likelihood(
        self, 
        action_samples: Tensor, 
        flow_matching_model: FlowMatchingModel, 
        global_cond: Tensor,
        generator: Optional[torch.Generator] = None
    ):
        """Compute log-likelihood of sampled action sequences."""
        scorer_ode_solver = ODESolver(flow_matching_model.unet)
        _, log_probs = scorer_ode_solver.sample_with_log_likelihood(
            x_init=action_samples,
            time_grid=self.lik_estimation_time_grid,
            global_cond=global_cond,
            log_p_0=self.gaussian_log_density,
            method=self.config.likelihood_ode_solver_cfg.method,
            atol=self.config.likelihood_ode_solver_cfg.atol,
            rtol=self.config.likelihood_ode_solver_cfg.rtol,
            exact_divergence=self.config.likelihood_ode_solver_cfg.exact_divergence,
            generator=generator,
        )

        return -log_probs

    def conditional_sample_with_recording(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """
        Sample an action sequence conditioned on an observation and record
        intermediate artifacts needed for uncertainty estimation.

        Args:
            observation: Info about the environment used to create the conditioning vector for
                the flow matching model. It has to contain the following items:
                {
                "observation.state": (B, n_obs_steps, state_dim)

                "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                    AND/OR
                "observation.environment_state": (B, environment_dim)
                }
            generator: PyTorch random number generator.

        Returns:
            - Action sequence drawn from the flow matching model.
              Shape: (horizon, action_dim).
        """
        step_data: Dict[str, Any] = {}
        
        # Draw flow matching model from the Laplace posterior
        laplace_model = draw_laplace_flow_matching_model(
            laplace_posterior=self.laplace_posterior,
            flow_matching_model=self.flow_matching_model,
            generator=generator
        )

        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors for the sampler and scorer models
        global_cond = self.flow_matching_model.prepare_global_conditioning(observation)
        ensemble_global_cond = self.ensemble_model.prepare_global_conditioning(observation)
        laplace_global_cond = laplace_model.prepare_global_conditioning(observation)
        step_data["obs_embedding"] = global_cond.squeeze(0).cpu().numpy()

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.config.num_uncertainty_sequences, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states = self.ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond.repeat(self.config.num_uncertainty_sequences, 1),
            method=self.flow_matching_config.ode_solver_method,
            atol=self.flow_matching_config.atol,
            rtol=self.flow_matching_config.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )
        step_data["ode_states"] = ode_states.cpu().numpy()

        action_candidates = ode_states[-1]  # (num_uncertainty_sequences, horizon, action_dim)
        step_data["action_pred"] = action_candidates.cpu().numpy()

        # Evaluate scorers' velocities on the final sampled action sequence
        ensemble_terminal_vels: list[Tensor] = []
        laplace_terminal_vels: list[Tensor] = []
        for time in self.config.terminal_vel_eval_times:
            time_batch = torch.full(
                (self.config.num_uncertainty_sequences,), time, device=self.device, dtype=self.dtype
            )
            ensemble_terminal_vels.append(self.ensemble_model.unet(
                action_candidates, time_batch, ensemble_global_cond.repeat(self.config.num_uncertainty_sequences, 1)
            ))
            laplace_terminal_vels.append(laplace_model.unet(
                action_candidates, time_batch, laplace_global_cond.repeat(self.config.num_uncertainty_sequences, 1)
            ))
        step_data["terminal_vel_eval_times"] = np.array(self.config.terminal_vel_eval_times)
        step_data["ensemble_terminal_vels"] = torch.stack(ensemble_terminal_vels, dim=0).cpu().numpy()
        step_data["laplace_terminal_vels"] = torch.stack(laplace_terminal_vels, dim=0).cpu().numpy()

        # Compute log-likelihood of sampled action sequences under ensemble and laplace model
        ensemble_log_likelihood = self.compute_log_likelihood(
            action_samples=action_candidates,
            flow_matching_model=self.ensemble_model,
            global_cond=ensemble_global_cond.repeat(self.config.num_uncertainty_sequences, 1),
            generator=generator,
        )
        step_data["ensemble_log_likelihood"] = ensemble_log_likelihood
        laplace_log_likelihood = self.compute_log_likelihood(
            action_samples=action_candidates,
            flow_matching_model=laplace_model,
            global_cond=laplace_global_cond.repeat(self.config.num_uncertainty_sequences, 1),
            generator=generator,
        )
        step_data["laplace_log_likelihood"] = laplace_log_likelihood

        # Pick one action sequence at random to return
        action_selection_idx = torch.randint(
            low=0,
            high=self.config.num_uncertainty_sequences,
            size=(1,),
            generator=generator,
            device=self.device
        ).item()
        action_sample = action_candidates[action_selection_idx : action_selection_idx+1]  # (1, horizon, action_dim)

        # Store data from this action generation step
        self.rollout_data.append(step_data)

        return action_sample
    
    def save_data(
        self,
        output_dir: str | Path,
        episode_metadata: Dict[str, Any],
    ) -> None:
        """
        Save the recorded data (episode metadata + rollout data) as a .pkl file.
        The filename is constructed using the episode index from episode_metadata.
        """
        episode_idx = episode_metadata.get("episode")
        if episode_idx is None:
            raise ValueError("episode_metadata must contain an 'episode' key.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        success_flag = "s" if episode_metadata["successful"] else "f"

        output_path = output_dir / f"episode_{success_flag}_{episode_idx:04d}.pkl"

        data = {
            "metadata": episode_metadata,
            "rollout": self.rollout_data,
        }

        with output_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Reset rollout data after saving
        self.rollout_data.clear()

        print(f"Saved FIPER data for episode {episode_idx} to {output_path}.")