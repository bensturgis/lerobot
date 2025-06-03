import torch

from abc import ABC, abstractmethod
from torch import nn, Tensor
from torch.distributions import Independent, Normal
from typing import Optional, Tuple

from lerobot.common.policies.flow_matching.configuration_flow_matching import (
    FlowMatchingConfig,
    ComposedLikSamplerConfig,
    CrossLikEnsembleSamplerConfig,
    LikSamplerConfig,
    EpsilonBallSamplerConfig,
)
from lerobot.common.policies.flow_matching.ode_solver import ODESolver
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

# TODO: Initialize ODE solver and gaussian_log_density during class initialization
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
        generator: Optional[torch.Generator] = None,
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
        self.num_action_seq_samples = num_action_seq_samples
        self.generator = generator
        self.horizon = self.flow_matching_cfg.horizon
        self.action_dim = self.flow_matching_cfg.action_feature.shape[0]
        # Store latest sampled action sequences and their uncertainty scores for logging
        self.latest_action_candidates = None
        self.latest_uncertainties = None

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

    @abstractmethod
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample `num_action_seq_samples` many action sequences and compute their
        uncertainty score according to some specific metric.

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].

        Returns:
            - Action sequences samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores. Shape: [num_action_seq_samples,]
        """
        pass


class CrossLikelihoodEnsembleSampler(FlowMatchingUncertaintySampler):
    """
    Samples action sequences from a “sampler” flow-matching model and evaluates their
    negative log-likelihood under a separate “scorer” flow-matching model to estimate
    epistemic uncertainty. A lower likelihood under the scorer indicates greater
    uncertainty about the sampled sequence.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: CrossLikEnsembleSamplerConfig,
        sampler_velocity_model: nn.Module,
        scorer_velocity_model: nn.Module,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
            sampler_velocity_model: The flow-matching network used to generate action sequences.
            scorer_velocity_model: The flow-matching network used to compute log-likelihoods of
                the sampled sequences.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=sampler_velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            generator=generator,
        )
        self.scorer_velocity_model = scorer_velocity_model
        self.exact_divergence = cfg.exact_divergence
        self.method_name = "cross_likelihood_ensemble"
        
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
        scorer_global_cond: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generates action sequences using the sampler model, then computes their log-likelihoods
        under the scorer model to produce uncertainty scores.

        Args:
            global_cond: Conditioning feature vector for the sampler model.
                Shape: [cond_dim,] or [1, cond_dim].
            scorer_global_cond: Conditioning feature vector for the scorer model.
                Shape: [cond_dim,] or [1, cond_dim].

        Returns:
            - sampled_action_seqs: Action sequences drawn from the sampler model.
                Shape: [num_action_seq_samples, horizon, action_dim].
            - uncertainty_scores: Negative log-likelihoods of the corresponding
              action sequence under the scorer model. Shape: [num_action_seq_samples,],       
        """
        # Adjust shape of conditioning vectors
        global_cond = self._prepare_conditioning(global_cond)
        scorer_global_cond = self._prepare_conditioning(scorer_global_cond)

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
            generator=self.generator,
        )

        # Noise distribution is an isotropic Gaussian
        gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(self.horizon, self.action_dim, device=device),
                scale = torch.ones(self.horizon, self.action_dim, device=device),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob

        # Solve ODE forward from noise to sample action sequences
        sampling_ode_solver = ODESolver(self.velocity_model)
        sampled_action_seqs = sampling_ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            step_size=self.flow_matching_cfg.ode_step_size,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
        )

        # Store sampled action sequences for logging
        self.latest_action_candidates = sampled_action_seqs

        # Compute log-likelihood of sampled action sequences in scorer model        
        scoring_ode_solver = ODESolver(self.scorer_velocity_model)
        _, log_probs = scoring_ode_solver.sample_with_log_likelihood(
            x_init=sampled_action_seqs,
            time_grid=torch.tensor([1.0, 0.0], device=device, dtype=dtype),
            global_cond=scorer_global_cond,
            log_p_0 = gaussian_log_density,
            method=self.flow_matching_cfg.ode_solver_method,
            step_size=self.flow_matching_cfg.ode_step_size,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            exact_divergence=self.exact_divergence,
            generator=self.generator,
        )

        # Use negative log-likelihood as uncertainty score
        uncertainty_scores = -log_probs
        
        # Store uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        return sampled_action_seqs, uncertainty_scores


class ComposedLikelihoodSampler(FlowMatchingUncertaintySampler):
    """
    Samples action sequences, composes them with a previously executed sequence segment, 
    and uses their negative log-likelihoods under the flow matching model 
    as uncertainty scores.

    The key idea is that if the composed sequences have a high likelihood, then the model
    successfully anticipated what is likely to happen next. This implies a good internal model 
    of the environment.
    """
    def __init__(
        self,
        flow_matching_cfg: FlowMatchingConfig,
        cfg: ComposedLikSamplerConfig,
        velocity_model: nn.Module,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            generator=generator,
        )
        self.exact_divergence = cfg.exact_divergence
        # Store the action sequence and conditioning vector from the previous action
        # sequence generation
        self.prev_action_sequence = None
        self.prev_global_cond = None
        self.method_name = "composed_likelihood"

    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Samples `num_action_seq_samples` many new action sequences and computes
        uncertainty scores by composing them with a previous action sequence and evaluating
        their negative log-likelihoods under the flow model.

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].

        Returns:
            - Action sequence samples. Shape: [num_action_seq_samples, horizon, action_dim].
            - Uncertainty scores given by negative log-likelihood of composed trajectories.
              Shape: [num_action_seq_samples,]
        """
        # Adjust shape of conditioning vector
        global_cond = self._prepare_conditioning(global_cond)
        if self.prev_global_cond is not None:
            self.prev_global_cond = self._prepare_conditioning(self.prev_global_cond)
        
        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        # Sample noise priors
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
            generator=self.generator,
        )

        # Noise distribution is an isotropic Gaussian
        gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(self.horizon, self.action_dim, device=device),
                scale = torch.ones(self.horizon, self.action_dim, device=device),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob

        # Solve ODE forward from noise to sample action sequences
        ode_solver = ODESolver(self.velocity_model)
        new_action_seqs = ode_solver.sample(
            x_0=noise_samples,
            global_cond=global_cond,
            step_size=self.flow_matching_cfg.ode_step_size,
            method=self.flow_matching_cfg.ode_solver_method,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
        )
        # Store sampled action sequences for logging
        self.latest_action_candidates = new_action_seqs

        # If no previous trajectory is stored, return placeholder uncertainties
        if self.prev_action_sequence is None:
            uncertainty_scores = torch.full(
                (self.num_action_seq_samples,),
                float('-inf'),
                dtype=dtype,
                device=device
            )
            # Store computed uncertainty scores for logging
            self.latest_uncertainties = uncertainty_scores

            return new_action_seqs, uncertainty_scores

        # Indices where to split and recompose the trajectory
        prev_action_seq_end = self.flow_matching_cfg.n_obs_steps - 1 + self.flow_matching_cfg.n_action_steps
        new_action_seqs_start = self.flow_matching_cfg.n_obs_steps - 1
        new_action_seqs_end = new_action_seqs_start + (self.horizon - prev_action_seq_end)
        
        # Repeat previous prefix to match batch dimension
        prev_action_sequence_duplicated = self.prev_action_sequence.expand(self.num_action_seq_samples, -1, -1)
        
        # Compose full action sequences from stored prefix and newly sampled action sequences
        composed_action_seqs = torch.cat([
            prev_action_sequence_duplicated[:, :prev_action_seq_end, :],
            new_action_seqs[:, new_action_seqs_start:new_action_seqs_end, :]
        ], dim=1)    

        # Compute log-likelihood of composed action sequences
        _, log_probs = ode_solver.sample_with_log_likelihood(
            x_init=composed_action_seqs,
            time_grid=torch.tensor([1.0, 0.0], device=device, dtype=dtype),
            global_cond=self.prev_global_cond,
            log_p_0 = gaussian_log_density,
            method=self.flow_matching_cfg.ode_solver_method,
            step_size=self.flow_matching_cfg.ode_step_size,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            exact_divergence=self.exact_divergence,
            generator=self.generator,
        )

        # Use negative log-likelihood as uncertainty score
        uncertainty_scores = -log_probs

        # Store computed uncertainty scores for logging
        self.latest_uncertainties = uncertainty_scores

        return new_action_seqs, uncertainty_scores
    

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
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            generator=generator,
        )
        self.exact_divergence = cfg.exact_divergence
        self.method_name = "likelihood"
    
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Samples `num_action_seq_samples` many action sequences x_1 and computes their
        log-likelihoods log(p_1(x_1)) via solving the combined flow matching ODE in
        forward direction. Then uses negative log-liklihoods -log(p_1(x_1)) as uncertainty
        scores such that a lower likelihood implies a larger uncertainty. 

        Args:
            global_cond: Single conditioning feature vector for the velocity
                model. Shape: [cond_dim,] or [1, cond_dim].

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

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        # Sample noise priors.
        noise_sample = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
            generator=self.generator,
        )

        # Noise distribution is an isotropic Gaussian.
        gaussian_log_density = Independent(
            Normal(
                loc = torch.zeros(self.horizon, self.action_dim, device=device),
                scale = torch.ones(self.horizon, self.action_dim, device=device),
            ),
            reinterpreted_batch_ndims=2
        ).log_prob
        
        # Solve combined flow matching ODE in forward direction to sample
        # action sequences x_1 and compute their log-likelihoods log(p_1(x_1)).
        ode_solver = ODESolver(self.velocity_model)
        action_seqs, log_probs = ode_solver.sample_with_log_likelihood(
            x_init=noise_sample,
            time_grid=torch.tensor([0.0, 1.0], device=device, dtype=dtype),
            global_cond=global_cond,
            log_p_0 = gaussian_log_density,
            method=self.flow_matching_cfg.ode_solver_method,
            step_size=self.flow_matching_cfg.ode_step_size,
            atol=self.flow_matching_cfg.atol,
            rtol=self.flow_matching_cfg.rtol,
            exact_divergence=self.exact_divergence,
            generator=self.generator,
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
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            cfg: Sampler-specific settings.
        """
        super().__init__(
            flow_matching_cfg=flow_matching_cfg,
            velocity_model=velocity_model,
            num_action_seq_samples=cfg.num_action_seq_samples,
            generator=generator,
        )
        self.epsilon = cfg.epsilon
        self.num_eps_ball_samples = cfg.num_eps_ball_samples
        self.method_name = "epsilon_ball"
    
    def conditional_sample_with_uncertainty(
        self,
        global_cond: Tensor,
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

        device = get_device_from_parameters(self.velocity_model)
        dtype = get_dtype_from_parameters(self.velocity_model)

        # Sample noise priors.
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
            generator=self.generator,
        )

        # Initialize tensors to store the action sequences and expansion factors.
        action_sequences = torch.zeros_like(noise_samples)
        expansion_factors = torch.zeros(
            size=(self.num_action_seq_samples,),
            dtype=dtype,
            device=device,
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
                device=device,
                dtype=dtype,
                generator=self.generator,
            )
            
            # Normalize each direction to unit length.
            norms = directions.view(self.num_eps_ball_samples, -1).norm(dim=1, keepdim=True)
            directions = directions / norms.view(self.num_eps_ball_samples, 1, 1)

            # Sample random radii so the points fill the ball uniformly.
            radii = torch.rand(
                self.num_eps_ball_samples,
                device=device,
                dtype=dtype,
                generator=self.generator,
            ) ** (1.0/(self.horizon*self.action_dim))
            radii = radii.view(self.num_eps_ball_samples, 1, 1) * self.epsilon                         # (N,1,1)
            
            # Compute average distance in noise space
            avg_noise_dist = radii.mean()

            # Get the samples in an epsilon-ball around the original noise sample.
            epsilon_samples = noise_sample + radii * directions   

            # Based on the center noise sample of the epsilon-ball use the velocity model
            # and an ODE solver to predict an action sequence sample from the target distribution.
            ode_solver = ODESolver(self.velocity_model)
            reference_action_sequence = ode_solver.sample(
                x_0=noise_sample.unsqueeze(0),
                global_cond=global_cond,
                step_size=self.flow_matching_cfg.ode_step_size,
                method=self.flow_matching_cfg.ode_solver_method,
                atol=self.flow_matching_cfg.atol,
                rtol=self.flow_matching_cfg.rtol,
            )
            action_sequences[action_seq_idx] = reference_action_sequence

            # Solve flow matching ODE for samples in epsilon-ball around the initial noise sample.
            perturbed_action_sequences = ode_solver.sample(
                x_0=epsilon_samples,
                global_cond=global_cond.repeat(self.num_eps_ball_samples, 1),
                step_size=self.flow_matching_cfg.ode_step_size,
                method=self.flow_matching_cfg.ode_solver_method,
                atol=self.flow_matching_cfg.atol,
                rtol=self.flow_matching_cfg.rtol,
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
        