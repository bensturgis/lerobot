import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

from ..configuration_flow_matching import FlowMatchingConfig
from ..modelling_flow_matching import FlowMatchingModel
from ..ode_solver import ADAPTIVE_SOLVERS, FIXED_STEP_SOLVERS, ODESolver
from .configuration_fiper_data_recorder import FiperDataRecorderConfig


class FiperDataRecorder:
    def __init__(
        self,
        config: FiperDataRecorderConfig,
        flow_matching_config: FlowMatchingConfig,
        flow_matching_model: FlowMatchingModel,
    ):
        self.config = config
        self.flow_matching_config = flow_matching_config
        self.flow_matching_model = flow_matching_model
        self.ode_solver = ODESolver(velocity_model=self.flow_matching_model.unet)

        self.horizon = self.flow_matching_config.horizon
        self.action_dim = self.flow_matching_config.action_feature.shape[0]
        self.device = get_device_from_parameters(self.flow_matching_model)
        self.dtype = get_dtype_from_parameters(self.flow_matching_model)

        # Build time grid for sampling according to ODE solver method and scoring metric
        if self.flow_matching_config.ode_solver_method in FIXED_STEP_SOLVERS:
            self.sampling_time_grid = self.ode_solver.make_sampling_time_grid(
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
        
        self.rollout_data: List[Dict[str, Any]] = []

    def conditional_sample_with_recording(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        step_data: Dict[str, Any] = {}
        
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.flow_matching_model.prepare_global_conditioning(observation)
        step_data["obs_embedding"] = global_cond.squeeze(0).cpu().numpy()
        
        # Adjust shape of conditioning vector to match batch size of noise samples
        global_cond = global_cond.repeat(self.config.num_uncertainty_sequences, 1)

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
            global_cond=global_cond,
            method=self.flow_matching_config.ode_solver_method,
            atol=self.flow_matching_config.atol,
            rtol=self.flow_matching_config.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )
        step_data["ode_states"] = ode_states.cpu().numpy()

        action_candidates = ode_states[-1]  # (num_uncertainty_sequences, horizon, action_dim)
        step_data["action_pred"] = action_candidates.cpu().numpy()

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

        output_path = output_dir / f"episode_{episode_idx:04d}.pkl"

        data = {
            "metadata": episode_metadata,
            "rollout": self.rollout_data,
        }

        with output_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved FIPER data for episode {episode_idx} to {output_path}.")