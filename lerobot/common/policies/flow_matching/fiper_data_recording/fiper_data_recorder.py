import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal

from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

from ..conditional_probability_path import (
    OTCondProbPath,
    VPDiffusionCondProbPath,
)
from ..configuration_flow_matching import FlowMatchingConfig
from ..modelling_flow_matching import FlowMatchingModel
from ..ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver,
    make_lik_estimation_time_grid,
    make_sampling_time_grid,
    select_ode_states,
)
from ..uncertainty.utils.laplace_utils import draw_laplace_flow_matching_model
from ..uncertainty.utils.sampler_utils import (
    compose_ode_states,
    select_and_expand_ode_states,
    splice_noise_with_prev,
)
from ..uncertainty.utils.scorer_artifacts import ScorerArtifacts
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
        extra_times = []
        if self.config.ode_eval_times is not None:
            extra_times = [t for t in self.config.ode_eval_times if 0.0 < t < 1.0]
        if self.flow_matching_config.ode_solver_method in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = make_sampling_time_grid(
                step_size=self.flow_matching_config.ode_step_size,
                extra_times=extra_times,
                device=self.device
            )
        elif self.flow_matching_config.ode_solver_method in ADAPTIVE_SOLVERS:
            self.sampling_time_grid = torch.tensor(
                [0.0, *extra_times, 1.0],
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

        # Select conditional probability path for computing the intermediate velocity difference scaling factors
        self.cond_vf_type = self.flow_matching_config.cond_vf_type
        if self.cond_vf_type == "vp":
            self.cond_prob_path = VPDiffusionCondProbPath(
                beta_min=self.flow_matching_config.beta_min,
                beta_max=self.flow_matching_config.beta_max,
            )
        elif self.cond_vf_type == "ot":
            self.cond_prob_path = OTCondProbPath()
        else:
            raise ValueError(
                f"Unknown conditional vector field type {self.cond_vf_type}."
            )
        
        # Store the conditioning vector and ODE states from the previous action
        # sequence generation
        self.prev_global_cond: Optional[Tensor] = None
        self.prev_ode_states: Optional[Tensor] = None

        # Index of the selected action sequence from the previous actions batch
        self.prev_selected_action_idx: Optional[int] = None

        # Store scorer flow matching model from the previous action sequence generation
        self.prev_laplace_model: Optional[FlowMatchingModel] = None
        
        # Store data from action generation steps across rollout
        self.rollout_data: List[Dict[str, Any]] = []

    def record_terminal_vels(
        self,
        action_samples: Tensor,
        composed_action_samples: Optional[Tensor],
        laplace_model: FlowMatchingModel,
        ensemble_global_cond: Tensor,
        laplace_global_cond: Tensor,
    ) -> Dict[str, Union[np.ndarray, Tensor]]:
        """
        Evaluate terminal velocities at configured times for ensemble/laplace on the
        current samples, and for sampler/ensemble/laplace on composed samples.
        Returns the recorded data in a dict.
        """
        # Store scorers' velocities on the final sampled action sequence
        ensemble_terminal_vels: List[Tensor] = []
        laplace_terminal_vels: List[Tensor] = []
        # Store sampler's velocities on composed action sequence
        composed_terminal_vels: List[Tensor] = []
        # Store scorers' velocities on composed action sequence
        composed_ensemble_terminal_vels: List[Tensor] = []
        composed_laplace_terminal_vels: List[Tensor] = []
        for time in self.config.terminal_vel_eval_times:
            time_batch = torch.full(
                (self.config.num_uncertainty_sequences,), time, device=self.device, dtype=self.dtype
            )
            ensemble_terminal_vels.append(self.ensemble_model.unet(
                action_samples, time_batch, ensemble_global_cond
            ))
            laplace_terminal_vels.append(laplace_model.unet(
                action_samples, time_batch, laplace_global_cond
            ))
            if self.prev_selected_action_idx is not None:
                composed_terminal_vels.append(self.flow_matching_model.unet(
                    composed_action_samples, time_batch, self.prev_global_cond
                ))
                composed_ensemble_terminal_vels.append(self.ensemble_model.unet(
                    composed_action_samples, time_batch, self.prev_global_cond
                ))
                composed_laplace_terminal_vels.append(self.prev_laplace_model.unet(
                    composed_action_samples, time_batch, self.prev_global_cond
                ))
            else:
                composed_terminal_vels.append(
                    torch.full_like(action_samples, float('nan'))
                )
                composed_ensemble_terminal_vels.append(
                    torch.full_like(action_samples, float('nan'))
                )
                composed_laplace_terminal_vels.append(
                    torch.full_like(action_samples, float('nan'))
                )
        return {
            "terminal_eval_times": np.asarray(self.config.terminal_vel_eval_times),
            "ensemble_terminal_velocities": torch.stack(ensemble_terminal_vels, dim=0).cpu(),
            "laplace_terminal_velocities": torch.stack(laplace_terminal_vels, dim=0).cpu(),
            "composed_terminal_velocities": torch.stack(composed_terminal_vels, dim=0).cpu(),
            "composed_ensemble_terminal_velocities": torch.stack(composed_ensemble_terminal_vels, dim=0).cpu(),
            "composed_laplace_terminal_velocities": torch.stack(composed_laplace_terminal_vels, dim=0).cpu(),
        }

    def compute_log_likelihood(
        self, 
        action_samples: Tensor, 
        flow_matching_model: FlowMatchingModel, 
        global_cond: Tensor,
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
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

        return log_probs
    
    def record_log_likelihoods(
        self,
        action_samples: Tensor,
        composed_action_samples: Optional[Tensor],
        laplace_model: FlowMatchingModel,
        ensemble_global_cond: Tensor,
        laplace_global_cond: Tensor,
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
            flow_matching_model=self.ensemble_model,
            global_cond=ensemble_global_cond,
            generator=generator,
        )
        laplace_log_likelihood = self.compute_log_likelihood(
            action_samples=action_samples,
            flow_matching_model=laplace_model,
            global_cond=laplace_global_cond,
            generator=generator,
        )
        # Compute log-likelihood of composed action sequence under sampler model
        if self.prev_selected_action_idx is not None:
            composed_log_likelihood = self.compute_log_likelihood(
                action_samples=composed_action_samples,
                flow_matching_model=self.flow_matching_model,
                global_cond=self.prev_global_cond,
                generator=generator,
            )
        else:
            composed_log_likelihood = torch.full((self.config.num_uncertainty_sequences,), float('nan'))
        # Compute log-likelihood of composed action sequence under ensemble and laplace model
        if self.prev_selected_action_idx is not None:
            composed_ensemble_log_likelihood = self.compute_log_likelihood(
                action_samples=composed_action_samples,
                flow_matching_model=self.ensemble_model,
                global_cond=self.prev_global_cond,
                generator=generator,
            )
            composed_laplace_log_likelihood = self.compute_log_likelihood(
                action_samples=composed_action_samples,
                flow_matching_model=self.prev_laplace_model,
                global_cond=self.prev_global_cond,
                generator=generator,
            )
        else:
            composed_ensemble_log_likelihood = torch.full((self.config.num_uncertainty_sequences,), float('nan'))
            composed_laplace_log_likelihood = torch.full((self.config.num_uncertainty_sequences,), float('nan'))
        
        return {
            "ensemble_log_likelihood": ensemble_log_likelihood.cpu(),
            "laplace_log_likelihood": laplace_log_likelihood.cpu(),
            "composed_log_likelihood": composed_log_likelihood.cpu(),
            "composed_ensemble_log_likelihood": composed_ensemble_log_likelihood.cpu(),
            "composed_laplace_log_likelihood": composed_laplace_log_likelihood.cpu(),
        }
    
    def record_inter_vel_diffs(
        self,
        ode_states: Tensor,
        composed_ode_states: Optional[Tensor],
        laplace_model: FlowMatchingModel,
        global_cond: Tensor,
        ensemble_global_cond: Tensor,
        laplace_global_cond: Tensor,
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
        if self.prev_selected_action_idx is not None:
            selected_prev_ode_states, _ = select_ode_states(
                time_grid=self.sampling_time_grid,
                ode_states=select_and_expand_ode_states(self.prev_ode_states, self.prev_selected_action_idx),
                requested_times=torch.tensor(self.config.ode_eval_times, device=self.device, dtype=self.dtype)
            )
            selected_composed_ode_states, _ = select_ode_states(
                time_grid=self.sampling_time_grid,
                ode_states=composed_ode_states,
                requested_times=torch.tensor(self.config.ode_eval_times, device=self.device, dtype=self.dtype)
            )

        # Compute velocities at intermediate ODE states for sampler, ensemble and laplace model
        sampler_vels: List[Tensor] = []
        ensemble_vels: List[Tensor] = []
        laplace_vels: List[Tensor] = []
        # Compute velocities at original and composed intermediate ODE states for sampler model
        prev_sampler_vels: List[Tensor] = []
        composed_sampler_vels: List[Tensor] = []
        # Compute velocities at original and composed intermediate ODE states for ensemble and laplace model
        composed_ensemble_vels: List[Tensor] = []
        composed_laplace_vels: List[Tensor] = []
        vel_diff_scaling_factors: List[float] = []
        for timestep, time in enumerate(self.config.ode_eval_times):
            ode_state = selected_ode_states[timestep]
            time_batch = torch.full(
                (self.config.num_uncertainty_sequences,), time, device=self.device, dtype=self.dtype
            )
            sampler_vels.append(
                self.flow_matching_model.unet(ode_state, time_batch, global_cond)
            )
            ensemble_vels.append(
                self.ensemble_model.unet(ode_state, time_batch, ensemble_global_cond)
            )
            laplace_vels.append(
                laplace_model.unet(ode_state, time_batch, laplace_global_cond)
            )
            if self.prev_selected_action_idx is not None:
                prev_ode_state = selected_prev_ode_states[timestep]
                composed_ode_state = selected_composed_ode_states[timestep]
                prev_sampler_vels.append(
                    self.flow_matching_model.unet(prev_ode_state, time_batch, self.prev_global_cond)
                )
                composed_sampler_vels.append(
                    self.flow_matching_model.unet(composed_ode_state, time_batch, self.prev_global_cond)
                )
                composed_ensemble_vels.append(
                    self.ensemble_model.unet(composed_ode_state, time_batch, self.prev_global_cond)
                )
                composed_laplace_vels.append(
                    self.prev_laplace_model.unet(composed_ode_state, time_batch, self.prev_global_cond)
                )
            else:
                prev_sampler_vels.append(
                    torch.full_like(ode_state, float('nan'))
                )
                composed_sampler_vels.append(
                    torch.full_like(ode_state, float('nan'))
                )
                composed_ensemble_vels.append(
                    torch.full_like(ode_state, float('nan'))
                )
                composed_laplace_vels.append(
                    torch.full_like(ode_state, float('nan'))
                )
            vel_diff_scaling_factors.append(self.cond_prob_path.get_vel_diff_scaling_factor(t=time))

        return {
            "ode_eval_times": np.asarray(self.config.ode_eval_times),
            "velocities": torch.stack(sampler_vels, dim=0).cpu(),
            "ensemble_velocities": torch.stack(ensemble_vels, dim=0).cpu(),
            "laplace_velocities": torch.stack(laplace_vels, dim=0).cpu(),
            "prev_velocities": torch.stack(prev_sampler_vels, dim=0).cpu(),
            "composed_velocities": torch.stack(composed_sampler_vels, dim=0).cpu(),
            "composed_ensemble_velocities": torch.stack(composed_ensemble_vels, dim=0).cpu(),
            "composed_laplace_velocities": torch.stack(composed_laplace_vels, dim=0).cpu(),
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
        step_data["obs_embedding"] = global_cond.squeeze(0).cpu()
        
        # Broadcast conditioning to match the number of action samples
        global_cond = global_cond.expand(self.config.num_uncertainty_sequences, -1)
        ensemble_global_cond = ensemble_global_cond.expand(self.config.num_uncertainty_sequences, -1)
        laplace_global_cond = laplace_global_cond.expand(self.config.num_uncertainty_sequences, -1)

        # Sample noise priors
        noise_sample = torch.randn(
            size=(self.config.num_uncertainty_sequences, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        if self.prev_selected_action_idx is not None:
            # Reuse overlapping segment of noise from the previously selected trajectory
            # so that the newly sampled noise remains consistent with already executed actions
            noise_sample = splice_noise_with_prev(
                new_noise_sample=noise_sample,
                prev_noise_sample=self.prev_ode_states[0, self.prev_selected_action_idx],
                flow_matching_cfg=self.flow_matching_config
            )

        # Solve ODE forward from noise to sample action sequences
        ode_states = self.ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond,
            method=self.flow_matching_config.ode_solver_method,
            atol=self.flow_matching_config.atol,
            rtol=self.flow_matching_config.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )
        action_candidates = ode_states[-1]  # (num_uncertainty_sequences, horizon, action_dim)
        step_data["action_pred"] = action_candidates.cpu()

        if self.prev_selected_action_idx is not None:
            # Compose full ODE states from stored previous and new action generation
            composed_ode_states = compose_ode_states(
                prev_ode_states=self.prev_ode_states[
                    :, self.prev_selected_action_idx:self.prev_selected_action_idx+1, :, :
                ],
                new_ode_states=ode_states,
                flow_matching_cfg=self.flow_matching_config,
            )
            composed_action_samples = composed_ode_states[-1] # (num_uncertainty_sequences, horizon, action_dim)
        else:
            composed_ode_states = None
            composed_action_samples = None

        # Record terminal velocities
        terminal_vels_data = self.record_terminal_vels(
            action_samples=action_candidates,
            composed_action_samples=composed_action_samples,
            laplace_model=laplace_model,
            ensemble_global_cond=ensemble_global_cond,
            laplace_global_cond=laplace_global_cond,
        )
        step_data.update(terminal_vels_data)

        # Record log-likelihoods
        log_likelihood_data = self.record_log_likelihoods(
            action_samples=action_candidates,
            composed_action_samples=composed_action_samples,
            laplace_model=laplace_model,
            ensemble_global_cond=ensemble_global_cond,
            laplace_global_cond=laplace_global_cond,
            generator=generator,
        )
        step_data.update(log_likelihood_data)

        # Record intermediate velocity differences
        inter_vel_diff_data = self.record_inter_vel_diffs(
            ode_states=ode_states,
            composed_ode_states=composed_ode_states,
            laplace_model=laplace_model,
            global_cond=global_cond,
            ensemble_global_cond=ensemble_global_cond,
            laplace_global_cond=laplace_global_cond,
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

        # Store conditioning vector, ODE states and selected action index from the previous sampling step
        self.prev_global_cond = global_cond
        self.prev_ode_states = ode_states
        self.prev_selected_action_idx = action_selection_idx
        self.prev_laplace_model = laplace_model

        # Store data from this action generation step
        self.rollout_data.append(step_data)

        return action_sample
    
    def reset(self):
        """
        Reset internal state to prepare for a new rollout.
        """
        # Clear stored conditioning vector, ODE states and selected action sequence from previous step
        self.prev_global_cond: Optional[Tensor] = None
        self.prev_ode_states: Optional[Tensor] = None
        self.prev_selected_action_idx: Optional[int] = None

        # Reset the previous laplace model
        self.prev_laplace_model: Optional[FlowMatchingModel] = None
        
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

        output_path = output_dir / f"episode_{success_flag}_{episode_idx:04d}.pkl"

        data = {
            "metadata": episode_metadata,
            "rollout": self.rollout_data,
        }

        with output_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.reset()

        print(f"Saved FIPER data for episode {episode_idx} to {output_path}.")