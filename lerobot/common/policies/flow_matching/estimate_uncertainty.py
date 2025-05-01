import torch

from torch import nn, Tensor

from lerobot.common.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.common.policies.flow_matching.ode_solver import ODESolver
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters

class FlowMatchingUncertaintyEstimator:
    def __init__(self, config: FlowMatchingConfig, velocity_model: nn.Module):
        """
        Args:
            config: Configuration object for Flow Matching settings.
            velocity_model: The learned flow matching velocity model.
        """
        self.config = config
        self.velocity_model = velocity_model

    def epsilon_ball_expansion(
        self,
        global_cond: Tensor,
        epsilon: float = 1e-3,
        num_samples: int = 1000
    ) -> float:
        """
        Estimate how epsilon‐ball perturbations around an initial noise sample expand through
        the flow model.

        Draws num_samples points uniformly in an  epsilon-ball of radius around an initial
        noise sample, runs the ODE solver from each point, and computes the ratio of the
        average output to the average input divergence.

        Args:
            global_cond: Single conditioning feature vector for the velocity model.
            epsilon: Radius of the input noise ball.
            num_samples: Number of samples to draw from epsilon-ball around initial
                         noise sample.

        Returns:
            Factor by how much epsilon-ball expands when moving it through the flow matching
            vector field.
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

        # Sample noise prior.
        noise_sample = torch.randn(
            size=(1, horizon, action_dim),
            dtype=dtype,
            device=device,
        )

        # Sample num_samples directions on the unit sphere.
        directions = torch.randn(num_samples, horizon, action_dim, device=device, dtype=dtype)
        
        # Normalize each direction to unit length.
        norms = directions.view(num_samples, -1).norm(dim=1, keepdim=True)
        directions = directions / norms.view(num_samples, 1, 1)

        # Sample random radii so the points fill the ball uniformly.
        radii = torch.rand(num_samples, device=device, dtype=dtype) ** (1.0/(horizon*action_dim))
        radii = radii.view(num_samples, 1, 1) * epsilon                         # (N,1,1)
        
        # Compute average distance in noise space
        avg_noise_dist = radii.mean()
        print(f"Average distance in noise space: {avg_noise_dist}")

        # Get the samples in an epsilon-ball around the original noise sample.
        epsilon_samples = noise_sample + radii * directions   

        # Based on the center noise sample of the epsilon-ball use the velocity model
        # and an ODE solver to predict an action sequence sample from the target distribution.
        ode_solver = ODESolver(self.velocity_model)
        reference_action_sequence = ode_solver.sample(
            x_0=noise_sample,
            global_cond=global_cond,
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )

        # Solve flow matching ODE for samples in epsilon-ball around the initial noise sample.
        perturbed_action_sequences = ode_solver.sample(
            x_0=epsilon_samples,
            global_cond=global_cond.repeat(num_samples, 1),
            step_size=self.config.ode_step_size,
            method=self.config.ode_solver_method,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )

        # Compute average distance in action space.
        action_sequence_dist = (perturbed_action_sequences - reference_action_sequence).norm(dim=1) 
        avg_action_sequence_dist = action_sequence_dist.mean()
        print(f"Average distance in action space: {avg_action_sequence_dist}")

        # Compute the factor by which the average distance of the epsilon-ball samples increases
        # from the noise to the action space.
        expansion_factor = avg_action_sequence_dist / avg_noise_dist
        
        print(f"Expansion factor: {expansion_factor}.")

        return expansion_factor
        