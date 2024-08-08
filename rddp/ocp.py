from typing import Tuple

import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State


class OptimalControlProblem:
    """A generic optimal control (reinforcement learning) problem.

    min J = ∑ₜ ℓ(xₜ, uₜ, t),
    s.t. xₜ₊₁ = f(xₜ, uₜ),
         x₀ ∼ p₀(x₀).

    where x is the state, u is the control action, and p₀ is the initial state
    distribution.
    """

    def __init__(self, env: PipelineEnv, num_steps: int, u_max: float = 1.0):
        """Initialize the optimal control problem.

        Args:
            env: The system to control (includes dynamics and reward/cost).
            num_steps: The number of time steps in the planning horizon.
            u_max: The maximum control input magnitude.
        """
        self.env = env
        self.num_steps = num_steps
        self.u_max = u_max

    def rollout(
        self, initial_state: State, control_tape: jnp.ndarray
    ) -> Tuple[jnp.ndarray, State]:
        """Simulate a rollout of the system under a fixed control tape.

        Args:
            initial_state: The initial state x₀.
            control_tape: The control sequence U = [u₀, u₁, ..., u_T₋₁].

        Returns:
            The total cost J(U | x₀)
            The state trajectory [x₀, x₁, ..., x_T].
        """

        def scan_fn(carry: Tuple, u: jnp.ndarray):
            x, cost = carry
            u = self.u_max * jnp.tanh(u / self.u_max)
            cost -= x.reward * (1 - x.done)
            x_next = self.env.step(x, u)
            return (x_next, cost), x

        (final_state, total_cost), state_trajectory = jax.lax.scan(
            scan_fn, (initial_state, 0.0), control_tape
        )

        # Account for the last state x_T
        state_trajectory = jax.tree.map(
            lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)]),
            state_trajectory,
            final_state,
        )
        total_cost -= final_state.reward * (1 - final_state.done)

        return total_cost, state_trajectory
