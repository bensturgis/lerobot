import torch

from abc import ABC, abstractmethod
from torch import nn, Tensor
from typing import Optional, Tuple

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.ode_solver import ODESolver
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

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

    
class EpsilonBallExpansion(FlowMatchingUncertaintySampler):
    """
    Samples action sequences and measures how an epsilon-ball around each initial noise
    sample expands through the flow.
    """
    def __init__(
        self,
        config: FlowMatchingConfig,
        velocity_model: nn.Module,
        num_action_seq_samples: int,
        epsilon: float = 1e-3,
        num_eps_ball_samples: int = 1000,
        generator: Optional[torch.Generator] = None,
    ):
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
            epsilon: Radius of the input noise ball.
            num_eps_ball_samples: Number of samples to draw from epsilon-ball around initial
                         noise samples.

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
        
        horizon = self.config.horizon
        action_dim = self.config.action_feature.shape[0]

        # Sample noise priors.
        noise_samples = torch.randn(
            size=(self.num_action_seq_samples, horizon, action_dim),
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
        
        # Generate an action sequence and uncertainty score for each of the `num_action_seq_samples`
        # noise samples.
        for action_seq_idx, noise_sample in enumerate(noise_samples):
            # Sample num_samples directions on the unit sphere.
            directions = torch.randn(
                self.num_eps_ball_samples,
                horizon,
                action_dim,
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
            ) ** (1.0/(horizon*action_dim))
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
        