import jax.numpy as jnp

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
