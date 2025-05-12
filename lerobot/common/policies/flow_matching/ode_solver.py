import itertools
import torch
import warnings

from torch import nn, Tensor
from torchdiffeq import odeint
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

FIXED_STEP_SOLVERS = {
    "euler",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
}
ADAPTIVE_SOLVERS = {
    "dopri8",
    "dopri5",
    "bosh3",
    "fehlberg2",
    "adaptive_heun",
}

class ODESolver():
    """
    Wrapper class for solving flow-matching-based ODEs using a velocity model.

    Supports sampling from a learned distribution and computing log-likelihoods 
    of a target sample via numerical ODE integration with torchdiffeq.
    """
    def __init__(self, velocity_model: nn.Module):
        self.velocity_model = velocity_model

    def sample(
        self,
        x_0: Tensor,
        global_cond: Tensor,
        method: str,
        step_size: Optional[float],
        atol: float,
        rtol: float,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
    ) -> Union[Tensor, Sequence[Tensor]]:
        """
        Solve the flow matching ODE with the conditioned velocity model.

        Args:
            x_0: Initial sample from source distribution. Shape: [batch_size, ...].
            global_cond: Global conditioning vector, encoding the robot's state and its visual
                observations. Shape: [batch_size, cond_dim].
            method: An ODE solver method supported by torchdiffeq. For a complete list, see torchdiffeq.
                Some examples are:
                - Fixed-step examples: "euler", "midpoint", "rk4", ... 
                - Adaptive examples: "dopri5", "bosh3", ...
            step_size: Size of an integration step only for fixed-step solvers. Ignored
                for adaptive solvers.
            atol, rtol: Absolute/relative error tolerances for accepting an adaptive solver step.
                Ignored for fixed-step solvers.
            time_grid: Times at which ODE is evluated. Integration runs from time_grid[0] to time_grid[-1].
                Must start at 0.0 and end at 1.0 for flow matching sampling. 
            return_intermediates: If True then return intermediate evaluation points according to time_grid.
            enable_grad: If True then compute gradients during sampling.

        Returns:
            Solely the solution of the ODE at time 1.0 when `return_intermediates` = False, otherwise all
            evaluation points specified in time_grid.
        """
        if time_grid[0] != 0.0 and time_grid[-1] == 1.0:
            raise ValueError(f"Time grid must start at 0.0 and end at 1.0. Got {time_grid}.")
        
        # Ensure all tensors are on the same device
        time_grid = time_grid.to(x_0.device)
        global_cond = global_cond.to(x_0.device)
        
        # Validate input shapes and solver parameters and build keyword arguments for odeint method
        ode_kwargs = self._validate_and_configure_solver(
            x_init=x_0,
            global_cond=global_cond,
            method=method,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
        )

        def velocity_field(t: Tensor, x: Tensor) -> Tensor:
            """
            Helper function defining the right-hand side of the flow matching ODE
            d/dt φ_t(x) = v_t(φ_t(x), global_cond). `global_cond` is captured from the
            outer scope.
            
            Args:
                t: Current scalar time of the ODE integration.
                x: Current state x_t along the flow trajectory. Shape like `x_0`.
            
            Returns:
                Velocity v_t(φ_t(x), global_cond) with the same shape as `x`.
            """
            return self.velocity_model(x, t.expand(x.shape[0]), global_cond)

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            trajetory = odeint(
                velocity_field,
                x_0,
                time_grid,
                method=method,
                **ode_kwargs,
            )

        return trajetory if return_intermediates else trajetory[-1]

    def sample_with_log_likelihood(
        self,
        x_init: Tensor,
        time_grid: Tensor,
        global_cond: Tensor,
        log_p_0: Callable[[Tensor], Tensor],
        method: str,
        step_size: Optional[float],
        atol: Optional[float],
        rtol: Optional[float],
        return_intermediates: bool = False,
        exact_divergence: bool = True,
        num_hutchinson_samples: Optional[int] = 3,
        generator: Optional[torch.Generator] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        """
        Integrate the combined flow matching ODE in either direction and return both the terminal
        sample and the log-likelihood log(p_1(x_1)) of the target sample x_1.

        This method supports both:
        - Forward integration from x_0 ~ p_0 using an increasing `time_grid` (0.0 to 1.0),
        which produces a target sample x_1 and computes log(p_1(x_1)).
        - Reverse integration from x_1 ~ p_1 using a decreasing `time_grid` (1.0 to 0.0),
        which reconstructs the corresponding x_0 ~ p_0 and computes log(p_1(x_1)).

        The integration direction is automatically inferred from the ordering of `time_grid`.

        Args:
            x_init: Initial state. Shape: [batch_size, ...].
                - If time_grid[0] = 0.0, sample from the source distribution, i.e. `x_init` ~ p_0. 
                - If time_grid[0] = 1.0, sample from the target distribution, i.e. `x_init` ~ p_1.
            time_grid: Times at which ODE is evluated. Integration runs from time_grid[0] to time_grid[-1].
                Use [0.0, ..., 1.0] for forward integration (starting from a source sample),
                or [1.0, ..., 0.0] for reverse integration (starting from a target sample).
            log_p_0: Log-probability function of the source distribution p_0.
            global_cond: Global conditioning vector, encoding the robot's state and its visual
                observations. Shape: [batch_size, cond_dim].
            method: An ODE solver method supported by torchdiffeq. For a complete list, see torchdiffeq.
                Some examples are:
                - Fixed-step examples: "euler", "midpoint", "rk4", ... 
                - Adaptive examples: "dopri5", "bosh3", ...
            step_size: Size of an integration step only for fixed-step solvers. Ignored
                for adaptive solvers.
            atol, rtol: Absolute/relative error tolerances for accepting an adaptive solver step.
                Ignored for fixed-step solvers.
            return_intermediates: If True then return intermediate evaluation points according to time_grid.
            exact_divergence: Whether to compute the exact divergence or estimate it using the Hutchinson
                trace estimator.
            num_hutchinson_samples: Number of Hutchinson samples to use when estimating the divergence. Higher
                values reduce variance but increase computation. Ignored if `exact_divergence=True`.
            generator : Pass a pre-seeded generator for reproducible results.

        Returns:
            - Either the terminal state (x_0 in forward and x_1 in backward direction) when
              `return_intermediates` = False or all evaluation points x_t specified in time_grid
              when `return_intermediates` = True
            - The estimated log-likelihood log(p_1(x_1)).
        """
        if not (
            (time_grid[0] == 0.0 and time_grid[-1] == 1.0) or
            (time_grid[0] == 1.0 and time_grid[-1] == 0.0)
        ):
            raise ValueError(f"Time grid must go from 0.0 to 1.0 or from 1.0 to 0.0. Got {time_grid}.")
        
        if not exact_divergence and num_hutchinson_samples is None:
            raise ValueError("`num_hutchinson_samples` must be specified when `exact_divergence` is False.")

        # Ensure all tensors are on the same device
        global_cond = global_cond.to(x_init.device)
        time_grid = time_grid.to(x_init.device)
        
        # Validate input shapes and solver parameters and build keyword arguments for odeint method
        ode_kwargs = self._validate_and_configure_solver(
            x_init=x_init,
            global_cond=global_cond,
            method=method,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
        )

        # Sample `num_hutchinson_samples` fixed Rademacher noise vector for which E[zz^T] = I
        # for the Hutchinson divergence estimator
        if not exact_divergence:
            z_samples = torch.randint(
                0, 2,
                (num_hutchinson_samples, *x_init.shape),
                device=x_init.device,
                dtype=x_init.dtype,
                generator=generator,
            ) * 2 - 1

        def velocity_field(t: Tensor, x: Tensor) -> Tensor:
            """
            Helper function defining the right-hand side of the flow matching ODE
            d/dt φ_t(x) = v_t(φ_t(x), global_cond). `global_cond` is captured
            from the outer scope. 
            
            Args:
                t: Current scalar time of the ODE integration.
                x: Current state x_t along the flow trajectory. Shape like `x_init`.
            
            Returns:
                Velocity v_t(φ_t(x), global_cond) with the same shape as `x`.
            """
            return self.velocity_model(x, t.expand(x.shape[0]), global_cond)

        def combined_dynamics(t: Tensor, states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
            """
            Helper function defining the right-hand side of the combined ODE to compute the
            flow trajectory as well as the log-likelihood of a target sample x_1 with
                d/dt φ_t(x) = v_t(φ_t(x), global_cond)
            and
                d/dt f(t) = div(v_t(φ_t(x), global_cond)) (exact divergence computation) or
                d/dt f(t) = z^T D_{x_t} v_t(x_t, global_cond) z (Hutchinson divergence estimator)
            For the log-likelihood computation the ODE is solved in forward or reverse direction.
            `global_cond` is captured from the outer scope. 

            Args:
                t: Current scalar time of the ODE integration.
                states: Current state x_t along the flow trajectory (`states[0]`) with shape like `x_1`
                    and integrated scalar divergence f(t) (`states[1]`).

            Returns:
                Velocity d/dt φ_t(x) with the same shape as `states[0]` and scalar divergence term d/dt f(t).
            """           
            # Current state φ_t(x) along the flow trajectory
            x_t = states[0]
            with torch.set_grad_enabled(True):
                x_t.requires_grad_()
                # Compute velocity v_t(φ_t(x), global_cond)
                v_t = velocity_field(t, x_t)

                # Compute or estimate divergence div(v_t(φ_t(x), global_cond))
                if exact_divergence:
                    # Compute exact divergence
                    # div(v_t(φ_t(x), global_cond)) = tr(D_{x_t} v_t(x_t, global_cond))
                    div = torch.zeros(x_t.shape[0], device=x_t.device)
                    feature_dims = v_t.shape[1:]
                    for idx in itertools.product(*(range(d) for d in feature_dims)):                      
                        # Add batch dimension to index
                        idx = [slice(None)] + list(idx)
                        div += torch.autograd.grad(
                            outputs=v_t[idx],
                            inputs=x_t,
                            grad_outputs=torch.ones_like(v_t[idx]).detach(),
                            retain_graph=True,
                        )[0][idx]
                else:
                    # Compute Hutchinson divergence estimator z^T D_x(v_t(x_t)) z by averaging
                    # over `num_hutchinson_samples` samples z
                    div = torch.zeros(x_t.shape[0], device=x_t.device)
                    for z in z_samples:                        
                        # Dot product v_t · z for each batch element
                        v_t_dot_z = torch.einsum(
                            "ij,ij->i", v_t.flatten(start_dim=1), z.flatten(start_dim=1)
                        )
                        # Gradient of v_t · z w.r.t. x_t
                        grad_v_t_dot_z = torch.autograd.grad(
                            outputs=v_t_dot_z,
                            inputs=x_t,
                            grad_outputs=torch.ones_like(v_t_dot_z).detach(),
                            retain_graph=True,
                        )[0]
                        # Single-probe divergence estimate: zᵀ ∇_x (v_t · z)
                        div += torch.einsum(
                            "ij,ij->i",
                            grad_v_t_dot_z.flatten(start_dim=1),
                            z.flatten(start_dim=1),
                        )

                    div *= (1.0 / num_hutchinson_samples)

            return v_t, div

        # Set initial state of the reverse-time combined ODE for initial noise sample
        # and log-likelihood computation
        initial_state = (x_init, torch.zeros(x_init.shape[0], device=x_init.device))

        # Solve the combined flow matching ODE to obtain a terminal state (x_1 for
        # forward direction and x_0 for reverse direction) and log-probability difference
        # of log(p_1(x_1)) and log(p_0(x_0))
        trajectory, log_prob_diff = odeint(
            combined_dynamics,
            initial_state,
            time_grid,
            method=method,
            **ode_kwargs,
        )

        if time_grid[-1] == 0:
            # Extract initial noise sample from reverse flow trajectory
            x_0 = trajectory[-1]
            
            # Compute log-probability of target sample using log-probability of initial noise
            # sample and change-of-variables correction
            log_p_1_x_1 = log_p_0(x_0) + log_prob_diff[-1]
        elif time_grid[-1] == 1:           
            # Compute log-probability of target sample using log-probability of initial noise
            # sample and change-of-variables correction
            log_p_1_x_1 = log_p_0(x_init) - log_prob_diff[-1]

        return (trajectory, log_p_1_x_1) if return_intermediates else (trajectory[-1], log_p_1_x_1)


    def _validate_and_configure_solver(
        self,
        x_init: Tensor,
        global_cond: Tensor,
        method: str,
        step_size: Optional[float],
        atol: Optional[float],
        rtol: Optional[float],
    ) ->  Dict[str, Any]:
        """
        Validate input shapes and solver parameters, and construct appropriate
        keyword arguments for `torchdiffeq.odeint`.
        """
        # Check input shapes
        if global_cond.dim() != 2:
            raise ValueError(
                f"`global_cond` must have dimensions (batch, cond_dim); "
                f"got tensor with shape {tuple(global_cond.shape)}."
            )

        if global_cond.shape[0] != x_init.shape[0]:
            raise ValueError(
                "`x_0` and `global_cond` must have same batch size:"
                f"`x_0` has {x_init.shape[0]} but global_cond has {global_cond.shape[0]}."
            )
        
        # Check ODE solver parameters
        if method in FIXED_STEP_SOLVERS:
            # Positive step‑size required
            if step_size is None or step_size <= 0.0:
                raise ValueError(
                    f"`step_size` must be a positive float for fixed-step solver '{method}'."
                )
            if atol is not None or rtol is not None:
                warnings.warn(
                    f"`atol`/`rtol` is ignored by fixed-step solver '{method}'."
                )
            return {"options": {"step_size": float(step_size)}}

        if method in ADAPTIVE_SOLVERS:
            if atol is None or rtol is None or atol <= 0.0 or rtol <= 0.0:
                raise ValueError(
                    f"`atol` and `rtol` must be positive for adaptive solver '{method}'."
                )
            if step_size is not None:
                raise warnings.warn(
                    f"`step_size` is ignored by adaptive solver '{method}'."
                )
            return {"atol": float(atol), "rtol": float(rtol)}

        raise ValueError(
            f"Unknown solver '{method}'. Choose one of "
            f"{sorted(FIXED_STEP_SOLVERS | ADAPTIVE_SOLVERS)}."
        )