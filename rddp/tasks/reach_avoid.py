import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.systems.single_integrator import SingleIntegrator
from rddp.tasks.base import OptimalControlProblem


class ReachAvoid(OptimalControlProblem):
    """A simple planar robot must reach a target while avoiding an obstacle."""

    def __init__(
        self,
        num_steps: int,
        obstacle_position: jnp.ndarray,
        target_position: jnp.ndarray,
    ):
        """Initialize the reach-avoid problem.

        Args:
            num_steps: The number of time steps T.
            obstacle_position: The position of the obstacle.
            target_position: The position of the target.
        """
        sys = SingleIntegrator(dt=1.0)
        super().__init__(sys, num_steps)

        self.obstacle_position = obstacle_position
        self.target_position = target_position

    def running_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: int
    ) -> jnp.ndarray:
        """The running cost function."""
        input_cost = 0.1 * u.dot(u)
        obstacle_avoidance_cost = jnp.exp(
            -1 / 0.3 * jnp.linalg.norm(x - self.obstacle_position) ** 2
        )
        return input_cost + obstacle_avoidance_cost

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost function."""
        err = x - self.target_position
        return err.dot(err)

    def plot_scenario(self) -> None:
        """Make a matplotlib visualization of the reach-avoid scenario."""
        # Green star at the target position.
        plt.plot(*self.target_position, "g*", markersize=20)

        # Red contour plot of the obstacle cost
        def obstacle_cost(px: jnp.ndarray, py: jnp.ndarray) -> jnp.ndarray:
            x = jnp.array([px, py])
            return self.running_cost(x, jnp.zeros(2), 0)

        x = jnp.linspace(-3, 3, 100)
        y = jnp.linspace(-3, 3, 100)
        X, Y = jnp.meshgrid(x, y)
        Z = jax.vmap(jax.vmap(obstacle_cost))(X, Y)
        plt.contourf(X, Y, Z, cmap="Reds", levels=100)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
