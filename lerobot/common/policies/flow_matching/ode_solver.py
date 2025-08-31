import itertools
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torchdiffeq import odeint

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
        atol: float,
        rtol: float,
        step_size: Optional[float] = None,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediate_states: bool = False,
        return_intermediate_vels: bool = False,
        enable_grad: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tensor]
    ]:
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
            time_grid: Times at which ODE is evaluated. Integration runs from time_grid[0] to time_grid[-1].
                Must start at 0.0 and end at 1.0 for flow matching sampling. 
            return_intermediate_states: If True then return intermediate evaluation points according to time_grid.
            return_intermediate_vels: If True then return velocities at intermediate evaluation points acoording
                to time_grid.
            enable_grad: If True then compute gradients during sampling.

        Returns:
            - The solution of the ODE at time 1.0 when return_intermediate_states = False, otherwise all
            evaluation points specified in time_grid.
            - If return_intermediate_vels = True, additionally the velocities at the intermediate evaluation
            points specified in time_grid.
        """
        if time_grid[0] != 0.0 or time_grid[-1] != 1.0:
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

        outputs: List[Tensor] = []

        if return_intermediate_states:
            outputs.append(trajetory)
        else:
            outputs.append(trajetory[-1])

        if return_intermediate_vels:
            velocities = []
            for t, x_t in zip(time_grid, trajetory, strict=False):
                t_batch = t.expand(x_t.shape[0])
                velocities.append(self.velocity_model(x_t, t_batch, global_cond))
            outputs.append(torch.stack(velocities, dim=0))

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def sample_with_log_likelihood(
        self,
        x_init: Tensor,
        time_grid: Tensor,
        global_cond: Tensor,
        log_p_0: Callable[[Tensor], Tensor],
        method: str,
        atol: Optional[float],
        rtol: Optional[float],
        step_size: Optional[float] = None,
        return_intermediate_states: bool = False,
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
            return_intermediate_states: If True then return intermediate evaluation points according to time_grid.
            exact_divergence: Whether to compute the exact divergence or estimate it using the Hutchinson
                trace estimator.
            num_hutchinson_samples: Number of Hutchinson samples to use when estimating the divergence. Higher
                values reduce variance but increase computation. Ignored if `exact_divergence=True`.
            generator : Pass a pre-seeded generator for reproducible results.

        Returns:
            - Either the terminal state (x_0 in forward and x_1 in backward direction) when
              `return_intermediate_states` = False or all evaluation points x_t specified in time_grid
              when `return_intermediate_states` = True
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

        return (trajectory, log_p_1_x_1) if return_intermediate_states else (trajectory[-1], log_p_1_x_1)
    

    def make_sampling_time_grid(
        self,
        step_size: float,
        device: torch.device,
        extra_times: Optional[Union[Tensor, Sequence]] = None,
    ) -> Tensor:
        """
        Build a time grid from 0.0 to 1.0 with fixed step_size, plus extra points.

        Args:
            step_size: Spacing between regular points.
            extra_times: Additional timepoints to include.
            device: The device on which to create the time grid.

        Returns:
            A time grid of unique, sorted times in [0.0, 1.0.]
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
            device=device,
        )

        # Ensure time grid ends with 1.0
        if time_grid[-1] < 1.0:
            time_grid = torch.cat([
                time_grid, torch.tensor([1.0], device=device)
            ])

        # Merge step size time grid with extra times and sort
        if extra_times is not None:
            time_grid = torch.cat([
                time_grid, torch.tensor(extra_times, device=device).clamp(0.0, 1.0)
            ])

        # Remove near-duplicates and sort
        time_grid, _ = torch.sort(time_grid)
        keep = torch.ones_like(time_grid, dtype=torch.bool)
        keep[1:] = torch.diff(time_grid) > 1e-4
        time_grid = time_grid[keep]

        if time_grid[0].item() != 0.0 or time_grid[-1].item() != 1.0:
            raise RuntimeError("Sampling time grid must start at 0.0 and end at 1.0.")

        return time_grid
    

    def select_ode_states(
        self, time_grid: Tensor, ode_states: Tensor, requested_times: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract the ODE states (and their timestamps) that correspond to a set of
        requested times.

        Args:
            time_grid: A tensor of time points at which the ODE was evaluated.
            ode_states: A tensor of ODE states corresponding to the time points in time_grid.
            requested_times: A tensor of times for which to extract the ODE states.

        Returns:
            A tuple containing:
                - A tensor of ODE states corresponding to the requested times.
                - A tensor of the corresponding time points from `time_grid`.
        """
        if time_grid.size(0) != ode_states.size(0):
            raise ValueError(
                f"`time_grid` and `ode_states` must have the same length; "
                f"got {time_grid.size(0)} and {ode_states.size(0)}."
            )
        # Map each requested time to its index in the time grid
        matched_indices = []
        for req_t in requested_times:
            # Locate entries equal to requested time (within tolerance)
            time_mask = torch.isclose(time_grid, req_t, atol=1e-5, rtol=0)
            match_count = int(time_mask.sum().item())
            if match_count == 0:
                raise ValueError(f"Requested time {req_t.item()} not found in time_grid")
            if match_count > 1:
                raise ValueError(
                    f"Requested time {req_t.item()} matched {match_count} entries in time_grid; "
                    "expected exactly one."
                )
            
            # Grab index of match
            index = time_mask.nonzero(as_tuple=True)[0].item()
            matched_indices.append(index)

        # Select only the ODE states and time points that correspond to those indices
        selected_ode_states = ode_states[matched_indices]
        selected_grid_times = time_grid[matched_indices]

        return selected_ode_states, selected_grid_times


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
            if atol is not None or rtol is not None:
                warnings.warn(
                    f"`atol`/`rtol` is ignored by fixed-step solver '{method}'."
                )
            if step_size is not None:
                if step_size <= 0.0:
                    raise ValueError(
                        f"`step_size` must be a positive float for fixed-step solver '{method}'."
                    )
                return {"options": {"step_size": float(step_size)}}
            else:
                return {}    

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