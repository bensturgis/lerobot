import torch
from torch import nn, Tensor
from torchdiffeq import odeint

class ODESolver():
    def __init__(self, velocity_model: nn.Module):
        self.velocity_model = velocity_model

    def sample(
        self,
        x_0: Tensor,
        global_cond: Tensor,
        step_size: float | None,
        method: str,
        atol: float,
        rtol: float,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
    ) -> Tensor:
        r"""Solve the flow matching ODE with the conditioned velocity model.

        Args:
            x_0: Sample from source distribution. Shape: [batch_size, ...].
            global_cond: Global conditioning vector, encoding the robot’s
                state and its visual observations. Shape: [batch_size, cond_dim].
            step_size: The step size.
            method: A method supported by torchdiffeq. Commonly used solvers are "euler", "dopri5",
                "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol: Absolute tolerance, used for adaptive step solvers.
            rtol: Relative tolerance, used for adaptive step solvers.
            time_grid: The process is solved in the interval [min(time_grid), max(time_grid)] and if step_size
                is None then time discretization is set by the time grid. May specify a descending time_grid to
                solve in the reverse direction.
            return_intermediates: If True then return intermediate time steps according to time_grid.
            enable_grad: Whether to compute gradients during sampling.

        Returns:
            The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """
        time_grid = time_grid.to(x_0.device)

        def ode_func(t, x):
            return self.velocity_model(x, t.expand(x.shape[0]), global_cond)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                ode_func,
                x_0,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]