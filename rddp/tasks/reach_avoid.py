from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.struct import dataclass, field

from rddp.systems.single_integrator import SingleIntegrator
from rddp.tasks.base import OptimalControlProblem


@dataclass
class ReachAvoidConfig:
    """Configuration for the reach-avoid problem.

    Attributes:
        obstacle_position: The position of the obstacle.
        target_position: The position of the target.
        horizontal_limits: The maximum x position of the robot.
        vertical_limits: The maximum y position of the robot.
    """

    obstacle_position: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0])
    )
    target_position: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 1.5])
    )
    horizontal_limits: Tuple[float, float] = (-3, 3)
    vertical_limits: Tuple[float, float] = (-3, 3)


class ReachAvoid(OptimalControlProblem):
    """A simple planar robot must reach a target while avoiding an obstacle."""

    def __init__(
        self,
        num_steps: int,
        config: ReachAvoidConfig = None,
    ):
        """Initialize the reach-avoid problem.

        Args:
            num_steps: The number of time steps T.
            config: The configuration of the reach-avoid problem.
        """
        sys = SingleIntegrator(dt=1.0)
        super().__init__(sys, num_steps)

        if config is None:
            config = ReachAvoidConfig()
        self.config = config

    def running_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: int
    ) -> jnp.ndarray:
        """The running cost function."""
        input_cost = 0.1 * u.dot(u)
        x_obs = self.config.obstacle_position
        obstacle_avoidance_cost = jnp.exp(
            -1 / 0.3 * jnp.linalg.norm(x - x_obs) ** 2
        )
        return input_cost + obstacle_avoidance_cost

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost function."""
        err = x - self.config.target_position
        return err.dot(err)

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample an initial state x₀ from the initial state distribution."""
        rng, x_rng, y_rng = jax.random.split(rng, 3)
        px = jax.random.uniform(
            x_rng,
            minval=self.config.horizontal_limits[0],
            maxval=self.config.horizontal_limits[1],
        )
        py = jax.random.uniform(
            y_rng,
            minval=self.config.vertical_limits[0],
            maxval=self.config.vertical_limits[1],
        )
        return jnp.array([px, py])

    def plot_scenario(self) -> None:
        """Make a matplotlib visualization of the reach-avoid scenario."""
        # Green star at the target position.
        plt.plot(*self.config.target_position, "g*", markersize=20)

        # Red contour plot of the obstacle cost
        def obstacle_cost(px: jnp.ndarray, py: jnp.ndarray) -> jnp.ndarray:
            x = jnp.array([px, py])
            return self.running_cost(x, jnp.zeros(2), 0)

        x = jnp.linspace(*self.config.horizontal_limits, 100)
        y = jnp.linspace(*self.config.vertical_limits, 100)
        X, Y = jnp.meshgrid(x, y)
        Z = jax.vmap(jax.vmap(obstacle_cost))(X, Y)
        plt.contourf(X, Y, Z, cmap="Reds", levels=100)

        plt.xlim(*self.config.horizontal_limits)
        plt.ylim(*self.config.vertical_limits)
