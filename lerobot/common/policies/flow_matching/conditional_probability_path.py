import abc

import torch
from torch import Tensor

class CondProbPath(abc.ABC):
    """
    Abstract base class for conditional probability paths used in flow matching.

    Defines the interface for sampling from a time-dependent conditional probability
    path and computing the corresponding conditional velocity field.
    """
    @abc.abstractmethod
    def sample(self, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample x_t from the conditional probability p_t(x|x_1).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def velocity(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute the conditional velocity vector u_t(x|x_1).
        """
        raise NotImplementedError


class OTCondProbPath(CondProbPath):
    """
    Optimal transport-based conditional path using linear interpolation.
    """
    def __init__(self, sigma_min: float = 0):
        super().__init__()
        self.sigma_min = sigma_min

    def sample(self, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample x_t from the conditional probability p_t(x|x_1) based on optimal
        transport interpolant.
        """
        # Sample noise prior x_0 ~ N(0, I).
        x_0 = torch.randn_like(
            x_1,
            dtype=x_1.dtype,
            device=x_1.device,
        )
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def velocity(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute velocity u_t(x|x_1) of the optimal transport conditional vector field.
        """
        return (x_1 - (1 - self.sigma_min) * x_t) / (1 - (1 - self.sigma_min) * t)
    

class VPDiffusionCondProbPath(CondProbPath):
    """
    Variance-Preserving Diffusion conditional path with standard linear beta schedule.
    """
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def sample(self, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample x_t from the conditional probability p_t(x|x_1) based on variance-
        preserving diffusion path.
        """
        # Convert t so that s = 0 means clean data and s = 1 means full noise as
        # usual in Diffusion
        s = 1.0 - t
        # Compute the cumulative noise amount at s
        T = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * (s ** 2)
        # Compute how much of the original data remains
        alpha = torch.exp(-0.5 * T)
        # Compute how much new noise is added
        sigma = torch.sqrt(1.0 - torch.exp(-T))
        # Mix original data with noise
        epsilon = torch.randn_like(
            x_1,
            dtype=x_1.dtype,
            device=x_1.device,
        )
        return alpha * x_1 + sigma * epsilon

    def velocity(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Compute velocity u_t(x|x_1) of the variance-preserving diffusion conditional
        vector field.
        """
        # Convert t so that s = 0 means clean data and s = 1 means full noise as
        # usual in Diffusion
        s = 1.0 - t
        # Compute the scheduling parameters for this s
        T = self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * (s ** 2)
        alpha = torch.exp(-0.5 * T)
        sigma = torch.sqrt(1.0 - torch.exp(-T))
        beta = self.beta_min + s * (self.beta_max - self.beta_min)
        # Compute the derivatives of alpha and sigma with respect to t
        d_alpha = 0.5 * beta * alpha
        d_sigma = -0.5 * beta * torch.exp(-T) / sigma
  
        return (d_sigma / sigma) * (x_t - alpha * x_1) + d_alpha * x_1