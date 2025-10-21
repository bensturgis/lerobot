import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.common.flow_matching.conditional_probability_path import (
    OTCondProbPath,
    VPDiffusionCondProbPath,
)
from lerobot.policies.common.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
    make_lik_estimation_time_grid,
    make_sampling_time_grid,
    select_ode_states,
)
from lerobot.uncertainty.uncertainty_samplers.utils import compose_ode_states
from lerobot.uncertainty.uncertainty_scoring.laplace_utils.posterior_builder import (
    sample_adapter_from_posterior,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import ScorerArtifacts

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
        flow_matching_adapter: BaseFlowMatchingAdapter,
        scorer_artifacts: ScorerArtifacts,
    ):
        self.config = config
        self.flow_matching_adapter = flow_matching_adapter
        self.ode_solver = ODESolver()

        self.horizon = flow_matching_adapter.horizon
        self.n_action_steps = flow_matching_adapter.n_action_steps
        self.n_obs_steps = flow_matching_adapter.n_obs_steps
        self.action_dim = flow_matching_adapter.action_dim
        self.device = flow_matching_adapter.device
        self.dtype = flow_matching_adapter.dtype

        # Extract scorer artifacts
        self.ensemble_adapter = scorer_artifacts.ensemble_adapter
        self.laplace_posterior = scorer_artifacts.laplace_posterior
        if self.ensemble_adapter is None:
            raise ValueError("Ensemble model is required for FIPER data recording.")
        # elif self.laplace_posterior is None:
        #     raise ValueError("Laplace posterior is required for FIPER data recording.")

        # Build time grid for sampling according to ODE solver method and scoring metric
        extra_times = []
        if self.config.ode_eval_times is not None:
            extra_times = [t for t in self.config.ode_eval_times if 0.0 < t < 1.0]
        self.ode_solver_config = flow_matching_adapter.ode_solver_config
        if self.ode_solver_config["solver_method"] in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = make_sampling_time_grid(
                step_size=self.ode_solver_config["step_size"],
                extra_times=extra_times,
                device=self.device,
                dtype=self.dtype
            )
        elif self.ode_solver_config["solver_method"] in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = torch.tensor(
                [0.0, *extra_times, 1.0],
                device=self.device, dtype=self.dtype
            )
        else:
            raise ValueError(f"Unknown ODE solver method: {self.ode_solver_config['solver_method']}.")

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
            dtype=self.dtype,
        )

        # Select conditional probability path for computing the intermediate velocity difference scaling factors
        self.cond_vf_config = flow_matching_adapter.cond_vf_config
        if self.cond_vf_config["type"] == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=self.cond_vf_config["beta_min"],
                beta_max=self.cond_vf_config["beta_max"],
            )
        elif self.cond_vf_config["type"] == "ot":
            self.cond_prob_path = OTCondProbPath(self.cond_vf_config["sigma_min"])
        else:
            raise ValueError(
                f"Unknown conditional vector field type {self.cond_vf_config['type']}."
            )

        # Store the velocity function and ODE states from the previous action sequence generation
        self.prev_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self.prev_ode_states: Tensor | None = None

        # Index of the selected action sequence from the previous actions batch
        self.prev_selected_action_idx: Optional[int] = None

        # Store the velocity function of the ensemble and laplace model from the previous action sequence generation
        self.prev_ensemble_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self.prev_laplace_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None

        # Store data from action generation steps across rollout
        self.rollout_data: List[Dict[str, Any]] = []

    def record_terminal_vels(
        self,
        action_samples: Tensor,
        composed_action_samples: Optional[Tensor],
        laplace_velocity_fn: Callable[[Tensor, Tensor], Tensor],
        ensemble_velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Dict[str, Union[np.ndarray, Tensor]]:
        """
        Evaluate terminal velocities at configured times for ensemble/laplace on the
        current samples, and for sampler/ensemble/laplace on composed samples.
        Returns the recorded data in a dict.
        """
        # Store scorers' velocities on the final sampled action sequence
        ensemble_terminal_vels: List[Tensor] = []
        # laplace_terminal_vels: List[Tensor] = []
        # Store sampler's velocities on composed action sequence
        # composed_terminal_vels: List[Tensor] = []
        # Store scorers' velocities on composed action sequence
        # composed_ensemble_terminal_vels: List[Tensor] = []
        # composed_laplace_terminal_vels: List[Tensor] = []
        for time in self.config.terminal_vel_eval_times:
            time_tensor = torch.tensor(time, device=self.device, dtype=self.dtype)
            ensemble_terminal_vels.append(ensemble_velocity_fn(
                x_t=action_samples, t=time_tensor,
            ))
            # laplace_terminal_vels.append(laplace_velocity_fn(
            #     x_t=action_samples, t=time_tensor
            # ))
            # if self.prev_selected_action_idx is not None:
            #     composed_terminal_vels.append(self.prev_velocity_fn(
            #         x_t=composed_action_samples, t=time_tensor
            #     ))
            #     composed_ensemble_terminal_vels.append(self.prev_ensemble_velocity_fn(
            #         x_t=composed_action_samples, t=time_tensor
            #     ))
            #     composed_laplace_terminal_vels.append(self.prev_laplace_velocity_fn(
            #         x_t=composed_action_samples, t=time_tensor
            #     ))
            # else:
            #     composed_terminal_vels.append(
            #         torch.full_like(action_samples, float('nan'))
            #     )
            #     composed_ensemble_terminal_vels.append(
            #         torch.full_like(action_samples, float('nan'))
            #     )
            #     composed_laplace_terminal_vels.append(
            #         torch.full_like(action_samples, float('nan'))
            #     )
        return {
            "terminal_eval_times": np.asarray(self.config.terminal_vel_eval_times),
            "ensemble_terminal_velocities": torch.stack(ensemble_terminal_vels, dim=0).detach().cpu().numpy(),
            # "laplace_terminal_velocities": torch.stack(laplace_terminal_vels, dim=0).detach().cpu().numpy(),
            # "composed_terminal_velocities": torch.stack(composed_terminal_vels, dim=0).detach().cpu().numpy(),
            # "composed_ensemble_terminal_velocities": torch.stack(composed_ensemble_terminal_vels, dim=0).detach().cpu().numpy(),
            # "composed_laplace_terminal_velocities": torch.stack(composed_laplace_terminal_vels, dim=0).detach().cpu().numpy(),
        }

    def compute_log_likelihood(
        self,
        action_samples: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """Compute log-likelihood of sampled action sequences."""
        _, log_probs = self.ode_solver.sample_with_log_likelihood(
            x_init=action_samples,
            time_grid=self.lik_estimation_time_grid,
            velocity_fn=velocity_fn,
            log_p_0=self.gaussian_log_density,
            method=self.config.likelihood_ode_solver_cfg.method,
            atol=self.config.likelihood_ode_solver_cfg.atol,
            rtol=self.config.likelihood_ode_solver_cfg.rtol,
            exact_divergence=self.config.likelihood_ode_solver_cfg.exact_divergence,
            generator=generator,
        )

        return log_probs

    def record_log_likelihoods(
        self,
        action_samples: Tensor,
        composed_action_samples: Optional[Tensor],
        laplace_velocity_fn: Callable[[Tensor, Tensor], Tensor],
        ensemble_velocity_fn: Callable[[Tensor, Tensor], Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """
        Compute log-likelihoods of the current samples under ensemble/laplace and,
        if a composed trajectory exists, log-likelihoods of composed samples under
        sampler/ensemble/laplace. Returns the recorded data in a dict.
        """
        # Compute log-likelihood of sampled action sequences under ensemble and laplace model
        ensemble_log_likelihood = self.compute_log_likelihood(
            action_samples=action_samples,
            velocity_fn=ensemble_velocity_fn,
            generator=generator,
        )
        laplace_log_likelihood = self.compute_log_likelihood(
            action_samples=action_samples,
            velocity_fn=laplace_velocity_fn,
            generator=generator,
        )
        # Compute log-likelihood of composed action sequence under sampler model
        if self.prev_selected_action_idx is not None:
            composed_log_likelihood = self.compute_log_likelihood(
                action_samples=composed_action_samples,
                velocity_fn=self.prev_velocity_fn,
                generator=generator,
            )
        else:
            composed_log_likelihood = torch.full((self.config.num_uncertainty_sequences,), float('nan'))
        # Compute log-likelihood of composed action sequence under ensemble and laplace model
        if self.prev_selected_action_idx is not None:
            composed_ensemble_log_likelihood = self.compute_log_likelihood(
                action_samples=composed_action_samples,
                velocity_fn=self.prev_ensemble_velocity_fn,
                generator=generator,
            )
            composed_laplace_log_likelihood = self.compute_log_likelihood(
                action_samples=composed_action_samples,
                velocity_fn=self.prev_laplace_velocity_fn,
                generator=generator,
            )
        else:
            composed_ensemble_log_likelihood = torch.full((self.config.num_uncertainty_sequences,), float('nan'))
            composed_laplace_log_likelihood = torch.full((self.config.num_uncertainty_sequences,), float('nan'))

        return {
            "ensemble_log_likelihood": ensemble_log_likelihood.detach().cpu().numpy(),
            "laplace_log_likelihood": laplace_log_likelihood.detach().cpu().numpy(),
            "composed_log_likelihood": composed_log_likelihood.detach().cpu().numpy(),
            "composed_ensemble_log_likelihood": composed_ensemble_log_likelihood.detach().cpu().numpy(),
            "composed_laplace_log_likelihood": composed_laplace_log_likelihood.detach().cpu().numpy(),
        }

    def record_inter_vel_diffs(
        self,
        ode_states: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        laplace_velocity_fn: Callable[[Tensor, Tensor], Tensor],
        ensemble_velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Dict[str, Union[np.ndarray, Tensor]]:
        """
        Select intermediate ODE states at configured times and evaluate velocities for
        sampler/ensemble/laplace. If available, also evaluate on previous/composed states.
        Returns the recorded data in a dict.
        """
        # Select the ODE states that correspond to the ODE evaluation times
        selected_ode_states, _ = select_ode_states(
            time_grid=self.sampling_time_grid,
            ode_states=ode_states,
            requested_times=torch.tensor(self.config.ode_eval_times, device=self.device, dtype=self.dtype)
        )

        # Compute velocities at intermediate ODE states for sampler, ensemble and laplace model
        sampler_vels: List[Tensor] = []
        ensemble_vels: List[Tensor] = []
        # laplace_vels: List[Tensor] = []

        vel_diff_scaling_factors: List[float] = []
        for timestep, time in enumerate(self.config.ode_eval_times):
            ode_state = selected_ode_states[timestep]
            time_tensor = torch.tensor(time, device=self.device, dtype=self.dtype)
            sampler_vels.append(
                velocity_fn(x_t=ode_state, t=time_tensor)
            )
            ensemble_vels.append(
                ensemble_velocity_fn(x_t=ode_state, t=time_tensor)
            )
            # laplace_vels.append(
            #     laplace_velocity_fn(x_t=ode_state, t=time_tensor)
            # )
            vel_diff_scaling_factors.append(self.cond_prob_path.get_vel_diff_scaling_factor(t=time))

        return {
            "ode_eval_times": np.asarray(self.config.ode_eval_times),
            "velocities": torch.stack(sampler_vels, dim=0).detach().cpu().numpy(),
            "ensemble_velocities": torch.stack(ensemble_vels, dim=0).detach().cpu().numpy(),
            # "laplace_velocities": torch.stack(laplace_vels, dim=0).detach().cpu().numpy(),
            "vel_diff_scaling": np.asarray(vel_diff_scaling_factors),
        }

    def conditional_sample_with_recording(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """
        Sample an action sequence conditioned on an observation and record
        intermediate artifacts needed for uncertainty estimation.

        Args:
            observation: Info about the environment used to create the conditioning for
                the flow matching model.
            generator: PyTorch random number generator.

        Returns:
            - Action sequence drawn from the flow matching model.
              Shape: (horizon, action_dim).
        """
        step_data: Dict[str, Any] = {}

        # Draw flow matching model from the Laplace posterior
        # laplace_adapter = sample_adapter_from_posterior(
        #     laplace_posterior=self.laplace_posterior,
        #     uncertainty_adapter=self.flow_matching_adapter,
        #     generator=generator
        # )

        # Build the velocity function for sampler and scorer model conditioned on the current observation
        conditioning = self.flow_matching_adapter.prepare_conditioning(observation, self.config.num_uncertainty_sequences)
        velocity_fn = self.flow_matching_adapter.make_velocity_fn(conditioning=conditioning)
        step_data["obs_embedding"] = self.flow_matching_adapter.prepare_fiper_obs_embedding(conditioning=conditioning)

        ensemble_conditioning = self.ensemble_adapter.prepare_conditioning(observation, self.config.num_uncertainty_sequences)
        ensemble_velocity_fn = self.ensemble_adapter.make_velocity_fn(conditioning=ensemble_conditioning)

        # laplace_conditioning = laplace_adapter.prepare_conditioning(observation, self.config.num_uncertainty_sequences)
        # laplace_velocity_fn = laplace_adapter.make_velocity_fn(conditioning=laplace_conditioning)
        laplace_velocity_fn = None

        # Sample noise priors
        noise_sample = self.flow_matching_adapter.sample_prior(
            num_samples=self.config.num_uncertainty_sequences,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states = self.ode_solver.sample(
            x_0=noise_sample,
            velocity_fn=velocity_fn,
            method=self.ode_solver_config["solver_method"],
            atol=self.ode_solver_config["atol"],
            rtol=self.ode_solver_config["rtol"],
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )
        action_candidates = ode_states[-1]  # (num_uncertainty_sequences, horizon, action_dim)
        step_data["action_pred"] = action_candidates.detach().cpu().numpy()

        # if self.prev_selected_action_idx is not None:
        #     # Compose full ODE states from stored previous and new action generation
        #     composed_ode_states = compose_ode_states(
        #         prev_ode_states=self.prev_ode_states[
        #             :, self.prev_selected_action_idx:self.prev_selected_action_idx+1, :, :
        #         ],
        #         new_ode_states=ode_states,
        #         horizon=self.horizon,
        #         n_action_steps=self.n_action_steps,
        #         n_obs_steps=self.n_obs_steps
        #     )
        #     composed_action_samples = composed_ode_states[-1] # (num_uncertainty_sequences, horizon, action_dim)
        # else:
        #     composed_ode_states = None
        #     composed_action_samples = None
        composed_ode_states = None
        composed_action_samples = None

        # Record terminal velocities
        if "mode_distance" in self.config.scoring_metrics:
            terminal_vels_data = self.record_terminal_vels(
                action_samples=action_candidates,
                composed_action_samples=composed_action_samples,
                laplace_velocity_fn=laplace_velocity_fn,
                ensemble_velocity_fn=ensemble_velocity_fn,
            )
            step_data.update(terminal_vels_data)

        # Record log-likelihoods
        if "likelihood" in self.config.scoring_metrics:
            log_likelihood_data = self.record_log_likelihoods(
                action_samples=action_candidates,
                composed_action_samples=composed_action_samples,
                laplace_velocity_fn=laplace_velocity_fn,
                ensemble_velocity_fn=ensemble_velocity_fn,
                generator=generator,
            )
            step_data.update(log_likelihood_data)

        # Record intermediate velocity differences
        if "inter_vel_diff" in self.config.scoring_metrics:
            inter_vel_diff_data = self.record_inter_vel_diffs(
                ode_states=ode_states,
                velocity_fn=velocity_fn,
                laplace_velocity_fn=laplace_velocity_fn,
                ensemble_velocity_fn=ensemble_velocity_fn,
            )
            step_data.update(inter_vel_diff_data)

        # Pick one action sequence at random to return
        action_selection_idx = torch.randint(
            low=0,
            high=self.config.num_uncertainty_sequences,
            size=(1,),
            generator=generator,
            device=self.device
        ).item()
        action_sample = action_candidates[action_selection_idx : action_selection_idx+1]  # (1, horizon, action_dim)

        # Store velocity functions, ODE states and selected action index from the previous sampling step
        self.prev_velocity_fn = velocity_fn
        self.prev_ode_states = ode_states
        self.prev_selected_action_idx = action_selection_idx
        self.prev_laplace_velocity_fn = laplace_velocity_fn
        self.prev_ensemble_velocity_fn = ensemble_velocity_fn

        # Store data from this action generation step
        self.rollout_data.append(step_data)

        return action_sample

    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        # Clear stored velocity functions, ODE states and selected action sequence from previous step
        self.prev_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self.prev_ode_states: Optional[Tensor] = None
        self.prev_selected_action_idx: Optional[int] = None
        self.prev_laplace_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self.prev_ensemble_velocity_fn: Callable[[Tensor, Tensor], Tensor] | None = None

        # Clear recorded rollout data
        self.rollout_data.clear()

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

        filename = f"episode_{success_flag}_{episode_idx:04d}"
        task = episode_metadata["task"]
        task_id = episode_metadata.get("task_id")
        if "libero" in task and task_id is not None:
            filename += f"_task{task_id:02d}"
        output_path = output_dir / (filename + ".pkl")

        data = {
            "metadata": episode_metadata,
            "rollout": self.rollout_data,
        }

        with output_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.reset()

        print(f"Saved FIPER data for episode {episode_idx} to {output_path}.")
