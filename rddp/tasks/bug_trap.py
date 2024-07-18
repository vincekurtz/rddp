from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.struct import dataclass, field

from rddp.systems.unicycle import Unicycle
from rddp.tasks.base import OptimalControlProblem


@dataclass
class BugTrapConfig:
    """Configuration for the bug_trap problem.

    Attributes:
        target_position: The position of the target.
        horizontal_limits: The maximum x position of the robot.
        vertical_limits: The maximum y position of the robot.
    """

    target_position: jnp.ndarray = field(
        default_factory=lambda: jnp.array([1.0, 0.0])
    )
    horizontal_limits: Tuple[float, float] = (-3, 3)
    vertical_limits: Tuple[float, float] = (-3, 3)


class BugTrap(OptimalControlProblem):
    """A planar robot with nonlinear dynamics must escape a U-shaped maze."""

    def __init__(
        self,
        num_steps: int,
        config: BugTrapConfig = None,
    ):
        """Initialize the bug_trap problem.

        Args:
            num_steps: The number of time steps T.
            config: The configuration of the bug_trap problem.
        """
        sys = Unicycle(dt=0.1)
        super().__init__(sys, num_steps)

        if config is None:
            config = BugTrapConfig()
        self.config = config

        # Set obstacle positions to define the maze
        self.obs_positions = jnp.array(
            [
                [-1.0, 1.0],
                [-0.5, 1.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [0.0, 0.0],
                [0.0, -0.5],
                [0.0, -1.0],
                [-0.5, -1.0],
                [-1.0, -1.0],
            ]
        )

    def obstacle_avoidance_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the obstacle avoidance cost."""

        def scan_fn(total_cost: float, obs_pos: jnp.ndarray):
            cost = jnp.exp(-(jnp.linalg.norm(x[0:2] - obs_pos) ** 2) / 0.2)
            return total_cost + cost, None

        total_cost, _ = jax.lax.scan(scan_fn, 0.0, self.obs_positions)

        return 10 * total_cost

    def running_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: int
    ) -> jnp.ndarray:
        """The running cost function."""
        input_cost = 0.1 * u.dot(u)
        obstacle_cost = self.obstacle_avoidance_cost(x)

        return input_cost + obstacle_cost

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost function."""
        err = x[0:2] - self.config.target_position
        target_cost = 10 * err.dot(err)
        obstacle_cost = self.obstacle_avoidance_cost(x)
        return target_cost + obstacle_cost

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Initial state is fixed."""
        return jnp.array([-1.0, 0.0, 0.0])

    def plot_scenario(self) -> None:
        """Make a matplotlib visualization of the reach-avoid scenario."""
        # Green star at the target position.
        plt.plot(*self.config.target_position, "g*", markersize=20)

        # Red contour plot of the obstacle cost
        def obstacle_cost(px: jnp.ndarray, py: jnp.ndarray) -> jnp.ndarray:
            x = jnp.array([px, py, 0.0])
            return self.obstacle_avoidance_cost(x)

        x = jnp.linspace(*self.config.horizontal_limits, 100)
        y = jnp.linspace(*self.config.vertical_limits, 100)
        X, Y = jnp.meshgrid(x, y)
        Z = jax.vmap(jax.vmap(obstacle_cost))(X, Y)
        plt.contourf(X, Y, Z, cmap="Reds", levels=100)

        plt.xlim(*self.config.horizontal_limits)
        plt.ylim(*self.config.vertical_limits)
