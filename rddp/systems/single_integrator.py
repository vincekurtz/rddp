from typing import Tuple

import jax.numpy as jnp

from rddp.systems.base import DynamicalSystem


class SingleIntegrator(DynamicalSystem):
    """A two-dimensional single integrator.

    This could represent a velocity-controlled holonomic robot, for example.
    """

    def __init__(self, dt: float = 0.1):
        """Initialize the single integrator.

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
        return (2,)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """The shape of the observation space."""
        return (2,)

    def f(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Forward dynamics function."""
        return x + self.dt * u

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Output function."""
        return x
