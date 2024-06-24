from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp

from rddp.systems.base import DynamicalSystem


class OptimalControlProblem(ABC):
    """A generic optimal control (reinforcement learning) problem.

    min J = ∑ₜ ℓ(xₜ, uₜ, t) + ℓ_f(x_T),
    s.t. xₜ₊₁ = f(xₜ, uₜ),
         x₀ ∼ p₀(x₀).

    where x is the state, u is the control action, and p₀ is the initial state
    distribution.
    """

    def __init__(self, sys: DynamicalSystem, num_steps: int):
        """Initialize the optimal control problem.

        Args:
            sys: The dynamical system to control.
            num_steps: The number of time steps T.
        """
        self.sys = sys
        self.num_steps = num_steps

    @abstractmethod
    def running_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: int
    ) -> jnp.ndarray:
        """The running cost function ℓ(xₜ, uₜ, t).

        Args:
            x: The current state xₜ.
            u: The control action uₜ to apply.
            t: The current time step.

        Returns:
            The running cost ℓ(xₜ, uₜ, t).
        """

    @abstractmethod
    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost function ℓ_f(x_T).

        Args:
            x: The final state x_T.

        Returns:
            The terminal cost ℓ_f(x_T).
        """

    def total_cost(
        self, control_tape: jnp.ndarray, x0: jnp.ndarray
    ) -> jnp.ndarray:
        """The total cost function J(U, x₀).

        Args:
            control_tape: The control tape U = [u₀, u₁, ..., u_T₋₁].
            x0: The initial state x₀.

        Returns:
            The total cost J.
        """

        def scan_fn(carry: Tuple, t: int):
            x, cost = carry
            u = control_tape[t]
            cost += self.running_cost(x, u, t)
            x_next = self.sys.f(x, u)
            return (x_next, cost), None

        (x, cost), _ = jax.lax.scan(
            scan_fn, (x0, 0.0), jnp.arange(self.num_steps)
        )
        cost += self.terminal_cost(x)
        return cost
