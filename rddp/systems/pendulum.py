from typing import Tuple

import jax.numpy as jnp

from rddp.systems.base import DynamicalSystem


class Pendulum(DynamicalSystem):
    """A simple inverted pendulum."""

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        gravity: float = 9.81,
        dt: float = 0.1,
    ):
        """Initialize the pendulum.

        Args:
            mass: The mass of the pendulum.
            length: The length of the pendulum.
            gravity: Acceleration due to gravity.
            dt: The time step of the system.
        """
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.dt = dt

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of the state space [theta, theta_dot]."""
        return (2,)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Shape of the action space [torque]."""
        return (1,)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of the observation space [cos, sin, theta_dot]."""
        return (3,)

    def f(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Forward dynamics function."""
        theta = x[0] - jnp.pi
        theta_dot = x[1]
        tau = u[0]
        mgl = self.mass * self.gravity * self.length
        ml2 = self.mass * self.length**2
        theta_ddot = (tau - mgl * jnp.sin(theta)) / ml2
        xdot = jnp.array([theta_dot, theta_ddot])
        return x + self.dt * xdot

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Output function."""
        theta = x[0]
        theta_dot = x[1]
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])
