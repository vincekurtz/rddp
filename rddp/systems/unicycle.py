from typing import Tuple

import jax.numpy as jnp

from rddp.systems.base import DynamicalSystem


class Unicycle(DynamicalSystem):
    """A simple non-holonomic robot that drives in the plane."""

    def __init__(self, dt: float = 0.1):
        """Initialize the system.

        Args:
            dt: The time step of the system.
        """
        self.dt = dt

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """The shape of the state space."""
        return (3,)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """The shape of the action space."""
        return (2,)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """The shape of the observation space."""
        return (3,)

    def f(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Forward dynamics function."""
        theta = x[2]
        v, w = u
        xdot = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), w])
        return x + self.dt * xdot

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Output function."""
        return x
