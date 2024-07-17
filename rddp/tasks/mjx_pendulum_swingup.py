import jax
import jax.numpy as jnp
from mujoco import mjx

from rddp.systems.mjx_pendulum import MjxPendulum
from rddp.tasks.base import OptimalControlProblem


class MjxPendulumSwingup(OptimalControlProblem):
    """An inverted pendulum must swing up to the upright position."""

    def __init__(
        self,
        num_steps: int,
    ):
        """Initialize the pendulum swing-up problem.

        Args:
            num_steps: The number of time steps T.
            config: The configuration of the pendulum swing-up problem.
        """
        sys = MjxPendulum()
        super().__init__(sys, num_steps)

    def _state_cost(self, x: mjx.Data) -> jnp.ndarray:
        """Penalty encouraging the pendulum to swing up."""
        y = self.sys.g(x)
        state_err = jnp.array([1.0, 0.0, 0.0]) - y
        return state_err.dot(state_err)

    def running_cost(self, x: mjx.Data, u: jnp.ndarray, t: int) -> jnp.ndarray:
        """The running cost function."""
        input_cost = 0.01 * u.dot(u)
        state_cost = 0.1 * self._state_cost(x)
        return input_cost + state_cost

    def terminal_cost(self, x: mjx.Data) -> jnp.ndarray:
        """The terminal cost function."""
        return 1.0 * self._state_cost(x)

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> mjx.Data:
        """Sample an initial state."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        pos = jax.random.uniform(
            pos_rng,
            minval=-1.5 * jnp.pi,
            maxval=1.5 * jnp.pi,
        )
        vel = jax.random.uniform(
            vel_rng,
            minval=-10,
            maxval=10,
        )
        data = self.sys.make_data()
        data = data.replace(qpos=jnp.array([pos]), qvel=jnp.array([vel]))
        return data
