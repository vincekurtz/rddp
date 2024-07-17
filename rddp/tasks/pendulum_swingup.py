from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.struct import dataclass

from rddp.systems.pendulum import Pendulum
from rddp.tasks.base import OptimalControlProblem


@dataclass
class PendulumSwingupConfig:
    """Configuration for the pendulum swing-up problem.

    Attributes:
        time_step: The time step of the system.
        position_limits: The maximum joint angle to sample over.
        velocity_limits: The maximum joint velocity to sample over.
        control_cost: The weight of the running control penalty.
        state_cost: The weight of the running state cost.
        terminal_cost: The weight of the terminal state cost.
    """

    time_step: float = 0.2
    position_limits: Tuple[float, float] = (-1.5 * jnp.pi, 1.5 * jnp.pi)
    velocity_limits: Tuple[float, float] = (-10, 10)
    control_cost: float = 0.01
    state_cost: float = 0.01
    terminal_cost: float = 1.0


class PendulumSwingup(OptimalControlProblem):
    """An inverted pendulum must swing up to the upright position."""

    def __init__(
        self,
        num_steps: int,
        config: PendulumSwingupConfig = None,
    ):
        """Initialize the pendulum swing-up problem.

        Args:
            num_steps: The number of time steps T.
            config: The configuration of the pendulum swing-up problem.
        """
        if config is None:
            config = PendulumSwingupConfig()
        self.config = config
        sys = Pendulum(dt=config.time_step)
        super().__init__(sys, num_steps)

    def _state_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """Penalty encouraging the pendulum to swing up."""
        # y = self.sys.g(x)
        # state_err = jnp.array([1.0, 0.0, 0.0]) - y
        state_err = x
        return state_err.dot(state_err)

    def running_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: int
    ) -> jnp.ndarray:
        """The running cost function."""
        input_cost = self.config.control_cost * u.dot(u)
        state_cost = self.config.state_cost * self._state_cost(x)
        return input_cost + state_cost

    def terminal_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """The terminal cost function."""
        return self.config.terminal_cost * self._state_cost(x)

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
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
        """Make a matplotlib visualization of the pendulum swing-up scenario."""
        theta = jnp.linspace(*self.config.position_limits, 20)
        theta_dot = jnp.linspace(*self.config.velocity_limits, 20)
        T, TD = jnp.meshgrid(theta, theta_dot)
        X = jnp.stack([T, TD], axis=-1)
        X_next = jax.vmap(jax.vmap(lambda x: self.sys.f(x, jnp.zeros(1))))(X)
        Xdot = X_next - X
        plt.quiver(T, TD, Xdot[:, :, 0], Xdot[:, :, 1], color="k")

        plt.xlim(self.config.position_limits)
        plt.ylim(self.config.velocity_limits)
        plt.xlabel("Position")
        plt.ylabel("Velocity")
