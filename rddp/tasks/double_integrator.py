from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.struct import dataclass, field

from rddp.systems.double_integrator import DoubleIntegrator
from rddp.tasks.base import OptimalControlProblem


@dataclass
class DoubleIntegratorConfig:
    """Configuration for the double integrator problem.

    Attributes:
        target_state: The state we want to drive the system to.
        position_limits: The maximum position variable x[0].
        velocity_limits: The maximum velocity variable x[1].
    """

    target_state: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0])
    )
    position_limits: Tuple[float, float] = (-3, 3)
    velocity_limits: Tuple[float, float] = (-3, 3)


class DoubleIntegratorProblem(OptimalControlProblem):
    """A simple double integrator must reach a target state."""

    def __init__(
        self,
        num_steps: int,
        config: DoubleIntegratorConfig = None,
    ):
        """Initialize the double integrator problem.

        Args:
            num_steps: The number of time steps T.
            config: The configuration of the double integrator problem.
        """
        sys = DoubleIntegrator(dt=1.0)
        super().__init__(sys, num_steps)

        if config is None:
            config = DoubleIntegratorConfig()
        self.config = config

    def running_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: int
    ) -> jnp.ndarray:
        """The running cost function."""
        input_cost = 0.1 * u.dot(u)
        target_state = self.config.target_state
        state_cost = 0.1 * (x - target_state).dot(x - target_state)
        return input_cost + state_cost

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost function."""
        target_state = self.config.target_state
        return 0.1 * (x - target_state).dot(x - target_state)

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> jax.Array:
        """Sample an initial state."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        pos = jax.random.uniform(
            pos_rng,
            minval=self.config.position_limits[0],
            maxval=self.config.position_limits[1],
        )
        vel = jax.random.uniform(
            vel_rng,
            minval=self.config.velocity_limits[0],
            maxval=self.config.velocity_limits[1],
        )
        return jnp.array([pos, vel])

    def plot_scenario(self) -> None:
        """Make a matplotlib visualization of the double integrator scenario."""
        # Green star at the target state
        plt.plot(*self.config.target_state, "g*", markersize=20)

        # Vector field showing the dynamics
        p = jnp.linspace(*self.config.position_limits, 20)
        v = jnp.linspace(*self.config.velocity_limits, 20)
        P, V = jnp.meshgrid(p, v)
        X = jnp.stack([P, V], axis=-1)
        X_next = jax.vmap(jax.vmap(lambda x: self.sys.f(x, jnp.zeros(1))))(X)
        Xdot = X_next - X
        plt.quiver(P, V, Xdot[:, :, 0], Xdot[:, :, 1], color="k")

        plt.xlim(self.config.position_limits)
        plt.ylim(self.config.velocity_limits)
        plt.xlabel("Position")
        plt.ylabel("Velocity")
