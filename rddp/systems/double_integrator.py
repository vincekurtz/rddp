from typing import Tuple

import jax.numpy as jnp

from rddp.systems.base import DynamicalSystem


class DoubleIntegrator(DynamicalSystem):
    """A one-dimensional double integrator."""

    def __init__(self, dt: float = 0.1):
        """Initialize the double integrator.

        Args:
            dt: The time step of the system.
        """
        self.dt = dt

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """The shape of the state space."""
        return (2,)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """The shape of the action space."""
        return (1,)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """The shape of the observation space."""
        return (2,)

    def f(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Forward dynamics function."""
        xdot = jnp.array([x[1], u[0]])
        return x + self.dt * xdot

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Output function."""
        return x
