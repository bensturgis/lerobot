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