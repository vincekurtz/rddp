from typing import Callable, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt

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

    def simulate_and_render(
        self,
        x0: jnp.ndarray,
        policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
        num_steps: int,
    ) -> None:
        """Simulate the system and make a matplotlib plot of the trajectory."""
        xs = jnp.zeros((num_steps, 2))
        xs = xs.at[0].set(x0)
        for t in range(num_steps - 1):
            x = xs[t]
            u = policy_fn(self.g(x))
            x = self.f(x, u)
            xs = xs.at[t + 1].set(x)
        plt.plot(xs[:, 0], xs[:, 1], "o-")
