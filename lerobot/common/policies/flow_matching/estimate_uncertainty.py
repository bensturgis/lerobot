import torch

from abc import ABC, abstractmethod
from torch import nn, Tensor
from torch.distributions import Independent, Normal
from typing import Optional, Tuple

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
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
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        num_action_seq_samples: int,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            config: Configuration object for Flow Matching settings.
            velocity_model: The learned flow matching velocity model.
            num_action_seq_samples: How many action sequences and corresponding
                uncertainty scores to sample.
        """
        self.config = config
        self.velocity_model = velocity_model
        self.num_action_seq_samples = num_action_seq_samples
        self.generator = generator
        self.horizon = self.config.horizon
        self.action_dim = self.config.action_feature.shape[0]

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

class ComposedActionSequenceLikelihood(FlowMatchingUncertaintySampler):
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
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        num_action_seq_samples: int = 1,
        exact_divergence: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            exact_divergence: Whether to compute the exact divergence or estimate it
                using the Hutchinson trace estimator when computing the log-likelihood
                for an action sequence sample.
        """
        super().__init__(
            config=config,
            velocity_model=velocity_model,
            num_action_seq_samples=num_action_seq_samples,
            generator=generator,
        )
        self.exact_divergence = exact_divergence
        # Store the action sequence and conditioning vector from the previous action
        # sequence generation
        self.prev_action_sequence = None
        self.prev_global_cond = None

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
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )

        # If no previous trajectory is stored, return placeholder uncertainties
        if self.prev_action_sequence is None:
            uncertainty_scores = torch.full(
                (self.num_action_seq_samples,),
                float('-inf'),
                dtype=dtype,
                device=device
            )
            return new_action_seqs, uncertainty_scores

        # Indices where to split and recompose the trajectory
        prev_action_seq_end = self.config.n_obs_steps - 1 + self.config.n_action_steps
        new_action_seqs_start = self.config.n_obs_steps - 1
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
            method=self.config.ode_solver_method,
            step_size=self.config.ode_step_size,
            atol=self.config.atol,
            rtol=self.config.rtol,
            exact_divergence=self.exact_divergence,
            generator=self.generator,
        )

        # Use negative log-likelihood as uncertainty score
        uncertainty_scores = -log_probs

        return new_action_seqs, uncertainty_scores
    
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


class ActionSequenceLikelihood(FlowMatchingUncertaintySampler):
    """
    Samples multiple action sequences x_1 and use their negative log-likelihoods
    -log(p_1(x_1)) under the flow matching mode as an uncertainty score.
    Smaller likelihood values p_1(x_1) correspond to a more uniform target
    distribution p_1(x_1) and therefore imply a greater uncertainty.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        num_action_seq_samples: int = 1,
        exact_divergence: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            exact_divergence: Whether to compute the exact divergence or estimate it
                using the Hutchinson trace estimator when computing the log-likelihoods
                log(p_1(x_1)) for the action sequnce samples x_1.
        """
        super().__init__(
            config=config,
            velocity_model=velocity_model,
            num_action_seq_samples=num_action_seq_samples,
            generator=generator,
        )
        self.exact_divergence = exact_divergence
    
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

        self.gaussian_log_density = self.gaussian_log_density.to(device)

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
            method=self.config.ode_solver_method,
            step_size=self.config.ode_step_size,
            atol=self.config.atol,
            rtol=self.config.rtol,
            exact_divergence=self.exact_divergence,
            generator=self.generator,
        )

        # Uncertainty score is given by -log(p_1(x_1))
        uncertainty_scores = -log_probs

        return action_seqs, uncertainty_scores

    
class EpsilonBallExpansion(FlowMatchingUncertaintySampler):
    """
    Samples action sequences and measures how an epsilon-ball around each initial noise
    sample expands through the flow.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        num_action_seq_samples: int = 1,
        epsilon: float = 1e-3,
        num_eps_ball_samples: int = 1000,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            epsilon: Radius of the input noise ball.
            num_eps_ball_samples: Number of samples to draw from epsilon-ball around initial
                noise samples.
        """
        super().__init__(
            config=config,
            velocity_model=velocity_model,
            num_action_seq_samples=num_action_seq_samples,
            generator=generator,
        )
        self.epsilon = epsilon
        self.num_eps_ball_samples = num_eps_ball_samples
    
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
                step_size=self.config.ode_step_size,
                method=self.config.ode_solver_method,
                atol=self.config.atol,
                rtol=self.config.rtol,
            )
            action_sequences[action_seq_idx] = reference_action_sequence

            # Solve flow matching ODE for samples in epsilon-ball around the initial noise sample.
            perturbed_action_sequences = ode_solver.sample(
                x_0=epsilon_samples,
                global_cond=global_cond.repeat(self.num_eps_ball_samples, 1),
                step_size=self.config.ode_step_size,
                method=self.config.ode_solver_method,
                atol=self.config.atol,
                rtol=self.config.rtol,
            )

            # Compute average distance in action space.
            action_sequence_dist = (perturbed_action_sequences - reference_action_sequence).norm(dim=1) 
            avg_action_sequence_dist = action_sequence_dist.mean()

            # Compute the factor by which the average distance of the epsilon-ball samples increases
            # from the noise to the action space.
            expansion_factor = avg_action_sequence_dist / avg_noise_dist

            expansion_factors[action_seq_idx] = expansion_factor
            
        return action_sequences, expansion_factors
        