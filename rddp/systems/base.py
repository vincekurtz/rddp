from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


class DynamicalSystem(ABC):
    """A discrete-time dynamical system.

        xₜ₊₁ = f(xₜ, uₜ),
        yₜ = g(xₜ),

    where x is the state, u is the control action, and y is the output.
    """

    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """The shape of the state space."""

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """The shape of the action space."""

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """The shape of the observation space."""

    @abstractmethod
    def f(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Forward dynamics function.

        Args:
            x: The current state xₜ.
            u: The control action uₜ to apply.

        Returns:
            The next state xₜ₊₁.
        """

    @abstractmethod
    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Output function.

        Args:
            x: The current state.

        Returns:
            The output measurement.
        """

    def simulate_and_render(
        self,
        x0: jnp.ndarray,
        policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
        num_steps: int,
    ) -> None:
        """Simulate the system and render the resulting trajectory.

        The rendering could be something simple, like printing to the console,
        or something complicated, like running an interactive mujoco simulation.

        Args:
            x0: The initial state.
            policy_fn: A function that maps observations to actions.
            num_steps: The number of time steps to simulate.
        """
        raise NotImplementedError

    def rollout(
        self, control_tape: jnp.ndarray, x0: jnp.ndarray
    ) -> jnp.ndarray:
        """Simulate the system forward in time given a control tape.

        Args:
            control_tape: The control tape U = [u₀, u₁, ..., u_T₋₁].
            x0: The initial state x₀.

        Returns:
            The state trajectory X = [x₁, ..., x_T].
        """

        def scan_fn(x: jnp.ndarray, u: jnp.ndarray):
            x_next = self.f(x, u)
            return x_next, x_next

        _, X = jax.lax.scan(scan_fn, x0, control_tape)
        return X
