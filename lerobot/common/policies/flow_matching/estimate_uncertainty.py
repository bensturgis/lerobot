import math
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from torch import nn, Tensor
from torch.distributions import Independent, Normal
from torch.utils.data import DataLoader
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.configuration_uncertainty_sampler import (
    ComposedCrossEnsembleSamplerConfig,
    ComposedCrossLaplaceSamplerConfig,
    ComposedSequenceSamplerConfig,
    CrossEnsembleSamplerConfig,
    CrossLaplaceSamplerConfig,
    LikelihoodODESolverConfig,
    LikSamplerConfig,
    EpsilonBallSamplerConfig,
)
from lerobot.common.policies.flow_matching.laplace_utils import (
    draw_laplace_flow_matching_model,
    get_laplace_posterior
)
from lerobot.common.policies.flow_matching.ode_solver import (
    ADAPTIVE_SOLVERS,
    FIXED_STEP_SOLVERS,
    ODESolver
)
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters


class FlowMatchingUncertaintySampler(ABC):
    """
    Abstract base class for uncertainty samplers that sample multiple action sequences
    and their per-sample uncertainty based on a Flow Matching model.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        velocity_model: nn.Module,
        num_action_seq_samples: int,
        scoring_metric: Optional[str] = None,
        velocity_eval_times: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            flow_matching_cfg: Shared configuration object for Flow Matching settings.
            velocity_model: The learned flow matching velocity model.
            num_action_seq_samples: How many action sequences and corresponding
                uncertainty scores to sample.
        """
        self.flow_matching_cfg = flow_matching_cfg
        self.velocity_model = velocity_model
        self.sampling_ode_solver = ODESolver(velocity_model)
        self.num_action_seq_samples = num_action_seq_samples
        self.horizon = self.flow_matching_cfg.horizon
        self.action_dim = self.flow_matching_cfg.action_feature.shape[0]
        self.device = get_device_from_parameters(velocity_model)
        self.dtype = get_dtype_from_parameters(velocity_model)
        # Noise distribution is an isotropic Gaussian
        self.gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(self.horizon, self.action_dim, device=self.device),
                scale = torch.ones(self.horizon, self.action_dim, device=self.device),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob
        # Store latest sampled action sequences and their uncertainty scores for logging
        self.latest_action_candidates = None
        self.latest_uncertainties = None
        # Build time grid for sampling according to ODE solver method and scoring metric
        if flow_matching_cfg.ode_solver_method in FIXED_STEP_SOLVERS:
            if scoring_metric in ["intermediate_vel_norm", "intermediate_vel_diff"]:
                self.sampling_time_grid = self._make_sampling_time_grid(
                    step_size=flow_matching_cfg.ode_step_size,
                    extra_times=velocity_eval_times
                )
            else:
                self.sampling_time_grid = self._make_sampling_time_grid(
                    step_size=flow_matching_cfg.ode_step_size
                )
        elif flow_matching_cfg.ode_solver_method in ADAPTIVE_SOLVERS:
            if scoring_metric in ["intermediate_vel_norm", "intermediate_vel_diff"]:
                self.sampling_time_grid = torch.tensor(
                    [0.0, *velocity_eval_times, 1.0], device=self.device, dtype=self.dtype
                )
            else:
                self.sampling_time_grid = torch.tensor(
                    [0.0, 1.0], device=self.device, dtype=self.dtype
                )

    def _prepare_conditioning(self, global_cond: Tensor) -> Tensor:
        """
        Reshape single global conditioning vector to (num_action_seq_samples, cond_dim).
        """
        if global_cond.ndim == 1:
            global_cond = global_cond.unsqueeze(0)
        if global_cond.ndim != 2 or global_cond.size(0) != 1:
            raise ValueError(
                f"Expected `global_cond` to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )
        # repeat batch‐dim
        return global_cond.repeat(self.num_action_seq_samples, 1)
    
    def compose_action_seqs(
        self,
        prev_action_seq: Tensor,
        new_action_seq: Tensor   
    ) -> Tensor:
        """
        Stitch together a complete candidate action sequence by keeping the prefix that
        has already been executed and appending the freshly sampled suffix.

        Args:
            prev_action_seq: Sequence collected during the previous sampling step.
                Shape: (batch_size, horizon, action_dim).
            new_action_seq: Newly generated action sequence.
                Shape: (batch_size, horizon, action_dim).

        Returns:
            The composed action sequence. Shape: (batch_size, horizon, action_dim).
        """
        # Indices where to split and recompose the trajectory
        prev_action_seq_end = (
            self.flow_matching_cfg.n_obs_steps - 1 + self.flow_matching_cfg.n_action_steps
        )
        new_action_seqs_start = self.flow_matching_cfg.n_obs_steps - 1
        new_action_seqs_end = new_action_seqs_start + (self.horizon - prev_action_seq_end)
        
        # Repeat previous prefix to match batch dimension
        prev_action_sequence_duplicated = prev_action_seq.expand(
            self.num_action_seq_samples, -1, -1
        )
        
        # Compose full action sequences from stored prefix and newly sampled action sequences
        composed_action_seq = torch.cat([
            prev_action_sequence_duplicated[:, :prev_action_seq_end, :],
            new_action_seq[:, new_action_seqs_start:new_action_seqs_end, :]
        ], dim=1)

        return composed_action_seq

    def _make_sampling_time_grid(
        self,
        step_size: float,
        extra_times: Optional[Sequence[float]] = None,
    ) -> Tensor:
        """
        Build a time grid from 0.0 to 1.0 with fixed step_size, plus extra points.

        Args:
            step_size: Spacing between regular points.
            extra_times: Additional timepoints to include.

        Returns:
            A time grid of unique, sorted times in [0.0, 1.0.
        """
        if not (0 < step_size <= 1.0):
            raise ValueError("step_size must be > 0 and <= 1.")

        # How many full steps of step_size fit into [0,1]
        n = math.floor(1.0 / step_size)

        # Regular grid from 0.0 to (n * step_size)
        time_grid = torch.linspace(
            0.0,
            n * step_size,
            steps=n + 1,
            device=self.device,
            dtype=self.dtype,
        )

        # Ensure time grid ends with 1.0
        if time_grid[-1] < 1.0:
            time_grid = torch.cat([
                time_grid, torch.tensor([1.0], device=self.device, dtype=self.dtype)
            ])

        # Merge step size time grid with extra times and sort
        if extra_times:
            time_grid = torch.cat([
                time_grid, torch.tensor(extra_times, device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
            ])
            time_grid, _ = torch.sort(torch.unique(time_grid))

        if time_grid[0].item() != 0.0 or time_grid[-1].item() != 1.0:
            raise RuntimeError("Sampling time grid must start at 0.0 and end at 1.0.")

        return time_grid
    
    def _make_fixed_lik_estimation_time_grid(
        self, direction: Literal["backward", "forward"]
    ) -> Tensor:
        """
        Create a time grid for ODE-based likelihood estiamtion.

        The time grid consists of a coarse segment of 10 points evenly spaced from 0.0
        up to 0.9  and a fine segment of 10 points evenly spaced from 0.93 up to 1.0.  

        When 'direction' is "forward" this returns an ascending tensor:
            [0.00, 0.10, …, 0.90, 0.93, …, 1.00]

        When 'direction' is "backward"` it returns the reverse:
            [1.00, …, 0.93, 0.90, …, 0.00]

        Args:
            direction:  
                - "forward": grid runs 0.0, …, 1.0  
                - "backward": grid runs 1.0, …, 0.0  

        Returns:
            A 1D time grid consisting of a coarse and fine segment in the order requested.
        """
        coarse = torch.linspace(0.0, 0.9,  steps=10, dtype=torch.float32)
        fine = torch.linspace(0.93, 1.0, steps=10, dtype=torch.float32)
        grid = torch.cat([coarse, fine])

        if direction == "forward":
            return grid
        else:
            return grid.flip(0)

    def _get_lik_estimation_time_grid(self) -> Tensor:
        """
        Build time grid to score samples according to ODE solver method and scoring metric.
        """
        if self.flow_matching_cfg.ode_solver_method in FIXED_STEP_SOLVERS:
            direction = "forward" if self.method_name == "likelihood" else "backward"
            lik_estimation_time_grid = self._make_fixed_lik_estimation_time_grid(direction)
        else:
            if self.method_name == "likelihood":
                lik_estimation_time_grid = torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype)
            else:
                lik_estimation_time_grid = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)

        return lik_estimation_time_grid


    @abstractmethod
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample `num_action_seq_samples` many action sequences and compute their
        uncertainty score according to some specific metric.

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.

        Returns:
            - Action sequences samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores. Shape: [num_action_seq_samples,]
        """
        pass

    def score_sample(
        self,
        scoring_metric: str,
        scorer_velocity_model: nn.Module,
        scorer_global_cond: Tensor,
        ode_states: Tensor,
        velocity_eval_times: Sequence[float],
        exact_divergence: bool,
        lik_ode_solver_cfg: LikelihoodODESolverConfig,
        sampler_global_cond: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """
        Compute an uncertainty score for a batch of action sequence samples
        generated by the sampler flow-matching model using a scorer flow matching
        model.

        Args:
            scoring_metric: Choice of scoring metric:
                - "intermediate_vel_norm": For all passed intermediate ODE states, evaluate
                the scorer's velocity field and compute the average L2 norm.
                - "terminal_vel_norm": Evaluate the scorer velocity only on the final sampled
                action sequence but at several times close to t=1 and compute the average L2 norm.
                - "intermediate_vel_diff": At each intermediate state compare the sampler and scorer
                velocities, ||v_sampler(x_t) - v_scorer(x_t)||, and average them.
                - "likelihood": Run a reverse-time ODE under the scorer model to compute the log-
                likelihood of the final sample; the score is the negative log-likelihood.
            scorer_velocity_model: Flow matching velocity model to compute the velocities field for
                scoring.
            scorer_global_cond: Conditioning vector used for the scorer model.
                Shape: (batch_size, cond_dim).
            ode_states: States produced by the forward ODE solver. Shape: (num_eval_points, batch_size,
                horizon, action_dim).
            velocity_eval_times: Times at which the velocity model is evaluated to compute velocity-based
                scoring metrics.
            sampler_global_cond: Conditioning vector that was used for the sampler's velocity model.
                Needed for "intermediate_vel_diff".
            exact_divergence: Whether to compute exact divergence in the reverse-time ODE.
                Needed for "likelihood".
            generator: PyTorch random number generator.

        Returns:
            Uncertainty scores per sample where larger values indicate higher uncertainty.
            Shape: (batch_size,).
        """
        if scoring_metric in [
            "intermediate_vel_norm", "intermediate_vel_diff"
        ]:
            # Map each configured evaluation time to its index in the solver's time grid
            matched_indices = []
            for eval_time in torch.tensor(velocity_eval_times, device=self.device, dtype=self.dtype):
                # Locate entries equal to eval_t (within tolerance)
                time_mask = torch.isclose(self.sampling_time_grid, eval_time, atol=1e-5, rtol=0)
                match_count = int(time_mask.sum().item())
                if match_count == 0:
                    raise ValueError(f"Evaluation time {eval_time.item()} not found in sampling_time_grid")
                if match_count > 1:
                    raise ValueError(f"Evaluation time {eval_time.item()} matched {match_count} entries in sampling_time_grid; expected exactly one.")
                
                # Grab index of match
                index = time_mask.nonzero(as_tuple=True)[0].item()
                matched_indices.append(index)

            # Select only the ODE states and time points that correspond to those indices
            selected_ode_states = ode_states[matched_indices]
            selected_grid_times = self.sampling_time_grid[matched_indices]

        # Compute uncertainty based on selected metric
        if scoring_metric == "intermediate_vel_norm":
            # Evaluate velocity field at each intermediate time point under scorer
            per_step_vel_norms = []
            for time, noisy_action_seq in zip(selected_grid_times, selected_ode_states):
                time_batch = torch.full(
                    (ode_states[-1].shape[0],), time, device=self.device, dtype=self.dtype
                )
                velocity = scorer_velocity_model(
                    noisy_action_seq,
                    time_batch,
                    scorer_global_cond,
                )
                # L2 norm across time and action dims gives per-sample velocity magnitude
                per_step_vel_norms.append(torch.norm(velocity, dim=(1, 2)))
            
            # Use average velocity norm as uncertainty score
            return torch.stack(per_step_vel_norms, dim=0).mean(dim=0)
        elif scoring_metric == "terminal_vel_norm":
            # The sampled action sequence corresponds to the final state of the ODE
            sampled_action_seq = ode_states[-1]
            # Evaluate velocity on the final sampled sequence at times close to t=1
            terminal_vel_norms = []
            for time in velocity_eval_times:
                time_batch = torch.full(
                    (sampled_action_seq.shape[0],), time, device=self.device, dtype=self.dtype
                )
                velocity = scorer_velocity_model(
                    sampled_action_seq,
                    time_batch,
                    scorer_global_cond,
                )
                # L2 norm across time and action dims gives velocity magnitude
                terminal_vel_norms.append(torch.norm(velocity, dim=(1, 2)))

            # Use average velocity norm as uncertainty score 
            return torch.stack(terminal_vel_norms, dim=0).mean(dim=0)
        elif scoring_metric == "intermediate_vel_diff":
            # Evaluate difference between sampler and scorer velocity field at each
            # intermediate time point
            per_step_vel_diff: List[Tensor] = []
            for time, intermediate_state in zip(selected_grid_times, selected_ode_states):
                time_batch = torch.full(
                    (ode_states[-1].shape[0],), time, device=self.device, dtype=self.dtype
                )
                sampler_velocity = self.velocity_model(
                    intermediate_state,
                    time_batch,
                    sampler_global_cond,
                )
                scorer_velocity = scorer_velocity_model(
                    intermediate_state,
                    time_batch,
                    scorer_global_cond,
                )
                velocity_difference = sampler_velocity - scorer_velocity
                # L2 norm across time and action dims gives magnitude of velocity difference
                per_step_vel_diff.append(torch.norm(velocity_difference, dim=(1, 2)))
            
            # Use average velocity difference as uncertainty score
            return torch.stack(per_step_vel_diff, dim=0).mean(dim=0)
        elif scoring_metric == "likelihood":            
            # Compute log-likelihood of sampled action sequences in scorer model    
            scoring_ode_solver = ODESolver(scorer_velocity_model)
            _, log_probs = scoring_ode_solver.sample_with_log_likelihood(
                x_init=ode_states[-1],
                time_grid=self.lik_estimation_time_grid,
                global_cond=scorer_global_cond,
                log_p_0=self.gaussian_log_density,
                method=lik_ode_solver_cfg.method,
                atol=lik_ode_solver_cfg.atol,
                rtol=lik_ode_solver_cfg.rtol,
                exact_divergence=exact_divergence,
                generator=generator,
            )

            # Use negative log-likelihood as uncertainty score
            return -log_probs
        else:
            raise ValueError(
                f"Unsupported scoring_metric '{scoring_metric}'. "
                "Expected one of: 'intermediate_vel_norm', 'terminal_vel_norm', "
                "'intermediate_vel_diff', 'likelihood'."
            )


class ComposedCrossLaplaceSampler(FlowMatchingUncertaintySampler):
    """
    Splices newly sampled action sequence tails onto the previously executed
    prefix and evaluates the full trajectories with a flow matching scorer
    sampled from a fitted Laplace posterior.

    The class therefore mixes
    - sequence composition from ComposedSequenceSampler and  
    - cross laplace epistemic scoring from CrossLaplaceSampler.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: ComposedCrossLaplaceSamplerConfig,
        flow_matching_model: nn.Module,
        laplace_calib_loader: DataLoader,
        laplace_path: Union[str, Path],
    ):
        """
        Initializes the composed sequence cross laplace sampler.
        
        Args:
            cfg: Sampler-specific settings.
            flow_matching_model: The full flow matching model including velocity and RGB encoder.
            laplace_calib_loader: DataLoader providing samples for fitting the Laplace
                approximation.
            laplace_path: Path to save or load the Laplace posterior.
        """
        # Use the MAP velocity network for sampling action sequences
        self.flow_matching_model = flow_matching_model
        velocity_model = self.flow_matching_model.unet
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            scoring_metric=cfg.scoring_metric,
            velocity_eval_times=cfg.velocity_eval_times,
        )
        self.method_name = "composed_cross_laplace"
        # Whether to compute exact divergence for log-likelihood
        self.exact_divergence = cfg.exact_divergence
        # Choice of scoring metric
        self.scoring_metric = cfg.scoring_metric
        if self.scoring_metric not in ("likelihood", "terminal_vel_norm"):
            raise ValueError(
                f"Unsupported scoring_metric '{self.scoring_metric}'. "
                "Expected one of: 'likelihood', 'terminal_vel_norm'."
            )
        # Configuration of ODE solver to score samples via a likelihood estimate
        self.lik_ode_solver_cfg = cfg.likelihood_ode_solver_cfg
        # Time grid used to estimate the log-likelihood
        self.lik_estimation_time_grid = self._get_lik_estimation_time_grid()
        # Times at which to evaluate the velocity field for the velocity based scoring metrics
        self.velocity_eval_times = cfg.velocity_eval_times
        # Store the action sequence, conditioning vector and laplace model
        # from the previous action sequence generation step
        self.prev_action_sequence = None
        self.prev_laplace_global_cond = None
        self.prev_laplace_model = None
        
        # Get the fitted Laplace posterior
        self.laplace_posterior = get_laplace_posterior(
            cfg=cfg,
            flow_matching_model=self.flow_matching_model,
            laplace_calib_loader=laplace_calib_loader,
            laplace_path=laplace_path,
        )

    def conditional_sample_with_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Composes previous and current action sequence and evaluates the result
        with a Laplace sampled flow matching scorer.
        
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
            - sampled_action_seqs: Action sequences drawn from the MAP model.
              Shape: [num_action_seq_samples, horizon, action_dim].
            - uncertainty_scores: Uncertainty scores where a higher value means more
                uncertain. Shape: [num_action_seq_samples,].      
        """
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.flow_matching_model.prepare_global_conditioning(observation) # (B, global_cond_dim)

        # Adjust shape of conditioning vector
        global_cond = self._prepare_conditioning(global_cond)

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        new_action_seq = self.sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            time_grid=self.sampling_time_grid,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
        )
        # Store sampled action sequences for logging
        self.latest_action_candidates = new_action_seq

        if self.prev_action_sequence is None:
            # If no previous trajectory is stored, return placeholder uncertainties
            uncertainty_scores = torch.full(
                (self.num_action_seq_samples,),
                float('-inf'),
                dtype=self.dtype,
                device=self.device
            )
        else:
            # Compose full action sequences from stored prefix and newly sampled
            # action sequences
            composed_action_seq = self.compose_action_seqs(
                prev_action_seq=self.prev_action_sequence,
                new_action_seq=new_action_seq  
            )

            # Compute uncertainty based on selected metric
            uncertainty_scores = self.score_sample(
                scoring_metric=self.scoring_metric,
                scorer_velocity_model=self.prev_laplace_model.unet,
                scorer_global_cond=self.prev_laplace_global_cond,
                ode_states=composed_action_seq.unsqueeze(0),
                velocity_eval_times=self.velocity_eval_times,
                exact_divergence=self.exact_divergence,
                lik_ode_solver_cfg=self.lik_ode_solver_cfg,
            )
        
        # Store uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        # Draw flow matching model from the Laplace posterior
        laplace_flow_matching_model = draw_laplace_flow_matching_model(
            laplace_posterior=self.laplace_posterior,
            flow_matching_model=self.flow_matching_model,
            generator=generator
        )

        # Store conditioning vector of the scoring model from the previous action sampling step
        laplace_global_cond = laplace_flow_matching_model.prepare_global_conditioning(observation)  # (B, global_cond_dim)
        self.prev_laplace_global_cond = self._prepare_conditioning(laplace_global_cond)
        self.prev_laplace_model = laplace_flow_matching_model

        return new_action_seq, uncertainty_scores


class CrossLaplaceSampler(FlowMatchingUncertaintySampler):
    """
    Estimates epistemic uncertainty of flow matching model by fitting a Laplace 
    approximation to a subset of weights. Action sequences are sampled from the MAP
    model and scored using models sampled from the Laplace posterior.

    The Laplace approximation is fit to the final layers of the velocity and/or image encoder.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: CrossLaplaceSamplerConfig,
        flow_matching_model: nn.Module,
        laplace_calib_loader: DataLoader,
        laplace_path: Union[str, Path],
    ):
        """
        Args:
            cfg: Sampler-specific settings.
            flow_matching_model: The full flow matching model including velocity and RGB encoder.
            laplace_calib_loader: DataLoader providing samples for fitting the Laplace
                approximation.
            laplace_path: Path to save or load the Laplace posterior.
        """
        # Use the MAP velocity network for sampling action sequences
        self.flow_matching_model = flow_matching_model
        velocity_model = self.flow_matching_model.unet
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            scoring_metric=cfg.scoring_metric,
            velocity_eval_times=cfg.velocity_eval_times,
        )
        self.method_name = "cross_laplace"
        # Whether to compute exact divergence for log-likelihood
        self.exact_divergence = cfg.exact_divergence
        # Choice of scoring metric
        self.scoring_metric = cfg.scoring_metric
        # Configuration of ODE solver to score samples via a likelihood estimate
        self.lik_ode_solver_cfg = cfg.likelihood_ode_solver_cfg
        # Time grid used to estimate the log-likelihood
        self.lik_estimation_time_grid = self._get_lik_estimation_time_grid()
        # Times at which to evaluate the velocity field for the velocity based scoring metrics
        self.velocity_eval_times = cfg.velocity_eval_times
        
        # Get the fitted Laplace posterior
        self.laplace_posterior = get_laplace_posterior(
            cfg=cfg,
            flow_matching_model=self.flow_matching_model,
            laplace_calib_loader=laplace_calib_loader,
            laplace_path=laplace_path,
        )

    def conditional_sample_with_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Generates action sequences using the MAP flow matching model, then scores these
        samples under a Laplace-sampled model to obtain epistemic uncertainty.

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
            - sampled_action_seqs: Action sequences drawn from the MAP model.
              Shape: [num_action_seq_samples, horizon, action_dim].
            - uncertainty_scores: Uncertainty scores where a higher value means more
                uncertain. Shape: [num_action_seq_samples,].      
        """
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.flow_matching_model.prepare_global_conditioning(observation) # (B, global_cond_dim)

        # Adjust shape of conditioning vector
        global_cond = self._prepare_conditioning(global_cond)
        laplace_global_cond = self._prepare_conditioning(laplace_global_cond)

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states = self.sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )

        # Store sampled action sequences for logging
        sampled_action_seqs = ode_states[-1]
        self.latest_action_candidates = sampled_action_seqs

        # Draw flow matching model from the Laplace posterior and create the flow matching conditioning vector
        laplace_flow_matching_model = draw_laplace_flow_matching_model(
            laplace_posterior=self.laplace_posterior,
            flow_matching_model=self.flow_matching_model,
            generator=generator
        )
        laplace_global_cond = laplace_flow_matching_model.prepare_global_conditioning(observation)  # (B, global_cond_dim)

        # Compute uncertainty based on selected metric
        uncertainty_scores = self.score_sample(
            scoring_metric=self.scoring_metric,
            scorer_velocity_model=laplace_flow_matching_model.unet,
            scorer_global_cond=laplace_global_cond,
            ode_states=ode_states,
            velocity_eval_times=self.velocity_eval_times,
            sampler_global_cond=global_cond,
            exact_divergence=self.exact_divergence,
            lik_ode_solver_cfg=self.lik_ode_solver_cfg
        )
        
        # Store uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        return sampled_action_seqs, uncertainty_scores


class ComposedCrossEnsembleSampler(FlowMatchingUncertaintySampler):
    """
    Splices newly sampled action sequence tails onto the previously executed
    prefix and evaluates the full trajectories with a flow matching scorer
    from an independent training.

    The class therefore mixes
    - sequence composition from ComposedSequenceSampler and  
    - cross ensemble epistemic scoring from CrossEnsembleSampler.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: ComposedCrossEnsembleSamplerConfig,
        sampler_flow_matching_model: nn.Module,
        scorer_flow_matching_model: nn.Module,
    ):
        """
        Initializes the composed sequence cross ensemble sampler.

        Args:
            cfg: Sampler-specific settings.
            sampler_flow_matching_model: The flow matching network used to generate action
                sequences.
            scorer_flow_matching_model: Model to score the composed action sequence.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=sampler_flow_matching_model.unet,
            num_action_seq_samples=cfg.num_action_seq_samples,
            scoring_metric=cfg.scoring_metric,
            velocity_eval_times=cfg.velocity_eval_times,
        )
        self.method_name = "composed_cross_ensemble"
        # Save models for sampling and scoring
        self.sampler_flow_matching_model = sampler_flow_matching_model
        self.scorer_flow_matching_model = scorer_flow_matching_model
        # Whether to compute exact divergence for log-likelihood
        self.exact_divergence = cfg.exact_divergence
        # Choice of scoring metric
        self.scoring_metric = cfg.scoring_metric
        if self.scoring_metric not in ("likelihood", "terminal_vel_norm"):
            raise ValueError(
                f"Unsupported scoring_metric '{self.scoring_metric}'. "
                "Expected one of: 'likelihood', 'terminal_vel_norm'."
            )
        # Configuration of ODE solver to score samples via a likelihood estimate
        self.lik_ode_solver_cfg = cfg.likelihood_ode_solver_cfg
        # Time grid used to estimate the log-likelihood
        self.lik_estimation_time_grid = self._get_lik_estimation_time_grid()
        # Times at which to evaluate the velocity field for the velocity based scoring metrics
        self.velocity_eval_times = cfg.velocity_eval_times
        
        # Store the action sequence and conditioning vector from the previous action
        # sequence generation
        self.prev_action_sequence = None
        self.prev_scorer_global_cond = None

    def conditional_sample_with_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Composes previous and current action sequence and evaluates the result
        with an independent flow matching scorer.

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
            - sampled_action_seqs: Action sequences drawn from the sampler model.
                Shape: [num_action_seq_samples, horizon, action_dim].
            - uncertainty_scores: Uncertainty scores where a higher value means more
                uncertain. Shape: [num_action_seq_samples,].       
        """
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.sampler_flow_matching_model.prepare_global_conditioning(observation) # (B, global_cond_dim)
        
        # Adjust shape of conditioning vectors
        global_cond = self._prepare_conditioning(global_cond)

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        new_action_seq = self.sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
        )
        # Store sampled action sequences for logging
        self.latest_action_candidates = new_action_seq

        if self.prev_action_sequence is None:
            # If no previous trajectory is stored, return placeholder uncertainties
            uncertainty_scores = torch.full(
                (self.num_action_seq_samples,),
                float('-inf'),
                dtype=self.dtype,
                device=self.device
            )
        else:
            # Compose full action sequences from stored prefix and newly sampled
            # action sequences
            composed_action_seq = self.compose_action_seqs(
                prev_action_seq=self.prev_action_sequence,
                new_action_seq=new_action_seq  
            )

            # Compute uncertainty based on selected metric
            uncertainty_scores = self.score_sample(
                scoring_metric=self.scoring_metric,
                scorer_velocity_model=self.scorer_flow_matching_model.unet,
                scorer_global_cond=self.prev_scorer_global_cond,
                ode_states=composed_action_seq.unsqueeze(0),
                velocity_eval_times=self.velocity_eval_times,
                exact_divergence=self.exact_divergence,
                lik_ode_solver_cfg=self.lik_ode_solver_cfg,
            )
        
        # Store computed uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        # Store conditioning vector of the scoring model from the previous action sampling step
        scorer_global_cond = self.scorer_flow_matching_model.prepare_global_conditioning(observation)  # (B, global_cond_dim)
        self.prev_scorer_global_cond = self._prepare_conditioning(scorer_global_cond)

        return new_action_seq, uncertainty_scores


class CrossEnsembleSampler(FlowMatchingUncertaintySampler):
    """
    Samples action sequences from a "sampler" flow-matching model and evaluates their
    uncertainty under a separately trained "scorer" flow-matching model. Uncertainty
    can be measured using several different metrics.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: CrossEnsembleSamplerConfig,
        sampler_flow_matching_model: nn.Module,
        scorer_flow_matching_model: nn.Module,
    ):
        """
        Initializes the cross ensemble sampler.

        Args:
            cfg: Sampler-specific settings.
            sampler_flow_matching_model: The flow matching network used to generate action sequences.
            scorer_flow_matching_model: Model to score sampled actions.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=sampler_flow_matching_model.unet,
            num_action_seq_samples=cfg.num_action_seq_samples,
            scoring_metric=cfg.scoring_metric,
            velocity_eval_times=cfg.velocity_eval_times,
        )
        self.method_name = "cross_ensemble"
        # Save models for sampling and scoring
        self.sampler_flow_matching_model = sampler_flow_matching_model
        self.scorer_flow_matching_model = scorer_flow_matching_model
        # Whether to compute exact divergence for log-likelihood
        self.exact_divergence = cfg.exact_divergence
        # Configuration of ODE solver to score samples via a likelihood estimate
        self.lik_ode_solver_cfg = cfg.likelihood_ode_solver_cfg
        # Time grid used to estimate the log-likelihood
        self.lik_estimation_time_grid = self._get_lik_estimation_time_grid()
        # Times at which to evaluate the velocity field for the velocity based scoring metrics
        self.velocity_eval_times = cfg.velocity_eval_times
        # Choice of scoring metric
        self.scoring_metric = cfg.scoring_metric
        
    def conditional_sample_with_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Samples candidate action sequences and evaluates uncertainty under separate
        scorer flow matching model using one of several metrics.

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
            - sampled_action_seqs: Action sequences drawn from the sampler model.
                Shape: [num_action_seq_samples, horizon, action_dim].
            - uncertainty_scores: Uncertainty scores where a higher value means more
                uncertain. Shape: [num_action_seq_samples,].       
        """
        # Encode image features and concatenate them all together along with the state vector
        # to create the flow matching conditioning vectors
        global_cond = self.sampler_flow_matching_model.prepare_global_conditioning(observation) # (B, global_cond_dim)
        scorer_global_cond = self.scorer_flow_matching_model.prepare_global_conditioning(observation)  # (B, global_cond_dim)
        
        # Adjust shape of conditioning vectors
        global_cond = self._prepare_conditioning(global_cond)
        scorer_global_cond = self._prepare_conditioning(scorer_global_cond)

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        ode_states = self.sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
            return_intermediate_states=True,
        )

        # Store sampled action sequences for logging
        sampled_action_seqs = ode_states[-1]
        self.latest_action_candidates = sampled_action_seqs

        # Compute uncertainty based on selected metric
        uncertainty_scores = self.score_sample(
            scoring_metric=self.scoring_metric,
            scorer_velocity_model=self.scorer_flow_matching_model.unet,
            scorer_global_cond=scorer_global_cond,
            ode_states=ode_states,
            velocity_eval_times=self.velocity_eval_times,
            sampler_global_cond=global_cond,
            exact_divergence=self.exact_divergence,
            lik_ode_solver_cfg=self.lik_ode_solver_cfg,
        )

        # Store uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        return sampled_action_seqs, uncertainty_scores


class ComposedSequenceSampler(FlowMatchingUncertaintySampler):
    """
    Samples action sequences, composes them with a previously executed sequence segment, 
    and evaluates their likelihood under the flow matching model.

    The key idea is that if the composed sequences have a high likelihood, then the model
    successfully anticipated what is likely to happen next. This implies a good internal model 
    of the environment.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: ComposedSequenceSamplerConfig,
        velocity_model: nn.Module,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            scoring_metric=cfg.scoring_metric,
            velocity_eval_times=cfg.velocity_eval_times,
        )
        self.method_name = "composed_sequence"
        # Whether to compute exact divergence for log-likelihood
        self.exact_divergence = cfg.exact_divergence
        self.lik_ode_solver_cfg = cfg.likelihood_ode_solver_cfg
        # Choice of scoring metric
        self.scoring_metric = cfg.scoring_metric
        if self.scoring_metric not in ("likelihood", "terminal_vel_norm"):
            raise ValueError(
                f"Unsupported scoring_metric '{self.scoring_metric}'. "
                "Expected one of: 'likelihood', 'terminal_vel_norm'."
            )
        # Configuration of ODE solver to score samples via a likelihood estimate
        self.lik_ode_solver_cfg = cfg.likelihood_ode_solver_cfg
        # Time grid used to estimate the log-likelihood
        self.lik_estimation_time_grid = self._get_lik_estimation_time_grid()
        # Times at which to evaluate the velocity field for the velocity based scoring metrics
        self.velocity_eval_times = cfg.velocity_eval_times
        # Store the action sequence and conditioning vector from the previous action
        # sequence generation
        self.prev_action_sequence = None
        self.prev_global_cond = None

    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Samples `num_action_seq_samples` many new action sequences and computes
        uncertainty scores by composing them with a previous action sequence and evaluating
        them under the flow model.

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.

        Returns:
            - Action sequence samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores where a higher value means more uncertain.
                Shape: [num_action_seq_samples,].
        """
        # Adjust shape of conditioning vector
        global_cond = self._prepare_conditioning(global_cond)

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Solve ODE forward from noise to sample action sequences
        new_action_seq = self.sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            time_grid=self.sampling_time_grid,
        )
        # Store sampled action sequences for logging
        self.latest_action_candidates = new_action_seq

        # If no previous trajectory is stored, return placeholder uncertainties
        if self.prev_action_sequence is None:
            uncertainty_scores = torch.full(
                (self.num_action_seq_samples,),
                float('-inf'),
                dtype=self.dtype,
                device=self.device
            )
        else:
            # Compose full action sequences from stored prefix and newly sampled action sequences
            composed_action_seq = self.compose_action_seqs(
                prev_action_seq=self.prev_action_sequence,
                new_action_seq=new_action_seq  
            )

            # Compute uncertainty based on selected metric
            uncertainty_scores = self.score_sample(
                scoring_metric=self.scoring_metric,
                scorer_velocity_model=self.velocity_model,
                scorer_global_cond=self.prev_global_cond,
                ode_states=composed_action_seq.unsqueeze(0),
                velocity_eval_times=self.velocity_eval_times,
                exact_divergence=self.exact_divergence,
                lik_ode_solver_cfg=self.lik_ode_solver_cfg,
            )      

        # Store computed uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        # Store conditioning vector of the scoring model from the previous action sampling step
        self.prev_global_cond = global_cond

        return new_action_seq, uncertainty_scores
    

class LikelihoodSampler(FlowMatchingUncertaintySampler):
    """
    Samples multiple action sequences x_1 and use their negative log-likelihoods
    -log(p_1(x_1)) under the flow matching mode as an uncertainty score.
    Smaller likelihood values p_1(x_1) correspond to a more uniform target
    distribution p_1(x_1) and therefore imply a greater uncertainty.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: LikSamplerConfig,
        velocity_model: nn.Module,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
        )
        self.method_name = "likelihood"
        self.exact_divergence = cfg.exact_divergence
        # Configuration of ODE solver to score samples via a likelihood estimate
        self.lik_ode_solver_cfg = cfg.likelihood_ode_solver_cfg
        # Time grid used to estimate the log-likelihood
        self.lik_estimation_time_grid = self._get_lik_estimation_time_grid()
    
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Samples `num_action_seq_samples` many action sequences x_1 and computes their
        log-likelihoods log(p_1(x_1)) via solving the combined flow matching ODE in
        forward direction. Then uses negative log-liklihoods -log(p_1(x_1)) as uncertainty
        scores such that a lower likelihood implies a larger uncertainty. 

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.

        Returns:
            - Action sequences samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores given by negative log-likelihood of the action sequnece
              samples. Shape: [num_action_seq_samples,]
        """
        if global_cond.dim() == 1: # shape = (cond_dim,)
            global_cond = global_cond.unsqueeze(0)                                   # (1, cond_dim)
            global_cond = global_cond.repeat(self.num_action_seq_samples, 1)
        elif global_cond.dim() == 2 and global_cond.size(0) == 1: # shape = (1, cond_dim)
            global_cond = global_cond.repeat(self.num_action_seq_samples, 1)
        else:
            raise ValueError(
                f"Expected global_cond to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )

        # Sample noise priors.
        noise_sample = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Noise distribution is an isotropic Gaussian.
        gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(self.horizon, self.action_dim, device=self.device),
                scale = torch.ones(self.horizon, self.action_dim, device=self.device),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob
        
        # Solve combined flow matching ODE in forward direction to sample
        # action sequences x_1 and compute their log-likelihoods log(p_1(x_1)).
        action_seqs, log_probs = self.sampling_ode_solver.sample_with_log_likelihood(
            x_init=noise_sample,
            time_grid=self.lik_estimation_time_grid,
            global_cond=global_cond,
            log_p_0=gaussian_log_density,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            exact_divergence=self.exact_divergence,
            generator=generator,
        )

        # Uncertainty score is given by -log(p_1(x_1))
        uncertainty_scores = -log_probs

        # Store sampled action sequences and uncertainty scores for logging
        self.latest_action_candidates = action_seqs
        self.latest_uncertainties = uncertainty_scores

        return action_seqs, uncertainty_scores

    
class EpsilonBallSampler(FlowMatchingUncertaintySampler):
    """
    Samples action sequences and measures how an epsilon-ball around each initial noise
    sample expands through the flow.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: EpsilonBallSamplerConfig,
        velocity_model: nn.Module,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
        )
        self.epsilon = cfg.epsilon
        self.num_eps_ball_samples = cfg.num_eps_ball_samples
        self.method_name = "epsilon_ball"
    
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
        generator: torch.Generator | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Samples `num_action_seq_samples` many action sequences and computes their
        uncertainty scores by estimating how epsilon-ball perturbations around their
        initial noise samples expand through the flow model.

        To compute the uncertainty scores, it draws num_samples points uniformly in an
        epsilon-ball around each initial noise sample, runs the ODE solver from each of
        these epsilon ball samples, and computes the ratio of the average output to the
        average input divergence.

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].
            generator: PyTorch random number generator.

        Returns:
            - Action sequences samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores given by epsilon-ball expansion factors.
              Shape: [num_action_seq_samples,]
        """
        if global_cond.dim() == 1: # shape = (cond_dim,)
            global_cond = global_cond.unsqueeze(0)     # (1, cond_dim)
        elif global_cond.dim() == 2 and global_cond.size(0) == 1: # shape = (1, cond_dim)
            pass
        else:
            raise ValueError(
                f"Expected global_cond to contain exactly one feature vector "
                f"(shape (cond_dim,) or (1,cond_dim)), but got shape {tuple(global_cond.shape)}"
            )

        # Sample noise priors.
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        # Initialize tensors to store the action sequences and expansion factors.
        action_sequences = torch.zeros_like(noise_samples)
        expansion_factors = torch.zeros(
            size=(self.num_action_seq_samples,),
            dtype=self.dtype,
            device=self.device,
        )
        
        # TODO: Remove for loop and parallelize code
        # Generate an action sequence and uncertainty score for each of the `num_action_seq_samples`
        # noise samples.
        for action_seq_idx, noise_sample in enumerate(noise_samples):
            # Sample num_samples directions on the unit sphere.
            directions = torch.randn(
                self.num_eps_ball_samples,
                self.horizon,
                self.action_dim,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            
            # Normalize each direction to unit length.
            norms = directions.view(self.num_eps_ball_samples, -1).norm(dim=1, keepdim=True)
            directions = directions / norms.view(self.num_eps_ball_samples, 1, 1)

            # Sample random radii so the points fill the ball uniformly.
            radii = torch.rand(
                self.num_eps_ball_samples,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            ) ** (1.0/(self.horizon*self.action_dim))
            radii = radii.view(self.num_eps_ball_samples, 1, 1) * self.epsilon                         # (N,1,1)
            
            # Compute average distance in noise space
            avg_noise_dist = radii.mean()

            # Get the samples in an epsilon-ball around the original noise sample.
            epsilon_samples = noise_sample + radii * directions   

            # Based on the center noise sample of the epsilon-ball use the velocity model
            # and an ODE solver to predict an action sequence sample from the target distribution.
            reference_action_sequence = self.sampling_ode_solver.sample(
                x_0=noise_sample.unsqueeze(0),
                global_cond=global_cond,
                method=self.flow_matching_cfg.ode_solver_method,
                atol=self.flow_matching_cfg.atol,
                rtol=self.flow_matching_cfg.rtol,
                time_grid=self.sampling_time_grid,
            )
            action_sequences[action_seq_idx] = reference_action_sequence

            # Solve flow matching ODE for samples in epsilon-ball around the initial noise sample.
            perturbed_action_sequences = self.sampling_ode_solver.sample(
                x_0=epsilon_samples,
                global_cond=global_cond.repeat(self.num_eps_ball_samples, 1),
                method=self.flow_matching_cfg.ode_solver_method,
                atol=self.flow_matching_cfg.atol,
                rtol=self.flow_matching_cfg.rtol,
                time_grid=self.sampling_time_grid,
            )

            # Compute average distance in action space.
            action_sequence_dist = (perturbed_action_sequences - reference_action_sequence).norm(dim=1) 
            avg_action_sequence_dist = action_sequence_dist.mean()

            # Compute the factor by which the average distance of the epsilon-ball samples increases
            # from the noise to the action space.
            expansion_factor = avg_action_sequence_dist / avg_noise_dist

            expansion_factors[action_seq_idx] = expansion_factor

            # Store sampled action sequences and uncertainty scores for logging
            self.latest_action_candidates = action_sequences
            self.latest_uncertainties = expansion_factors
            
        return action_sequences, expansion_factors
        