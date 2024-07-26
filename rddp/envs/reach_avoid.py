import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax import base
from brax.envs.base import PipelineEnv, State


class ReachAvoidEnv(PipelineEnv):
    """A simple planar robot must reach a target while avoiding an obstacle."""

    def __init__(self):
        """Initialize the reach-avoid problem."""
        # Set problem parameters
        self.obstacle_position = jnp.array([0.0, 0.0])
        self.target_position = jnp.array([0.0, 1.5])
        self.horizontal_limits = (-3, 3)
        self.vertical_limits = (-3, 3)

        # super().__init__(**kwargs)

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Reset the environment to a new initial state.

        Args:
            rng: The random seed

        Returns:
            The initial state of the environment.
        """
        rng, x_rng, y_rng = jax.random.split(rng, 3)
        px = jax.random.uniform(
            x_rng,
            minval=self.horizontal_limits[0],
            maxval=self.horizontal_limits[1],
        )
        py = jax.random.uniform(
            y_rng,
            minval=self.vertical_limits[0],
            maxval=self.vertical_limits[1],
        )
        pipeline_state = base.State(
            q=jnp.array([px, py]),
            qd=jnp.zeros(2),
            x=base.Transform.create(pos=jnp.zeros(3)),
            xd=base.Motion.create(vel=jnp.zeros(3)),
            contact=None,
        )
        obs = jnp.zeros(2)
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Forward dynamics step.

        Args:
            state: container for the state (position) of the robot.
            action: the action (velocity) of the robot.

        Return:
            the next state, including the updated reward and observation.
        """
        # Forward dynamics
        pos = state.pipeline_state.q + action
        vel = action
        new_pipeline_state = state.pipeline_state.replace(q=pos, qd=vel)

        # Observation
        obs = pos

        # Reward
        action_cost = 0.1 * action.dot(action)
        obstacle_cost = self._obstacle_cost(pos)
        goal_cost = (pos - self.target_position).dot(pos - self.target_position)
        reward = -(action_cost + obstacle_cost + goal_cost)

        return state.replace(
            pipeline_state=new_pipeline_state, obs=obs, reward=reward
        )

    def _obstacle_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        """Cost associated with being close to the obstacle.

        Args:
            x: The position of the robot.

        Returns:
            The obstacle-related cost at the given position.
        """
        return jnp.exp(-3 * jnp.linalg.norm(x - self.obstacle_position) ** 2)

    @property
    def action_size(self) -> int:
        """Actions are velocities in R^2."""
        return 2

    @property
    def observation_size(self) -> int:
        """Observations are positions in R^2."""
        return 2

    def plot_scenario(self) -> None:
        """Make a matplotlib visualization of the reach-avoid scenario."""
        # Green star at the target position.
        plt.plot(*self.target_position, "g*", markersize=20)

        # Red contour plot of the obstacle cost
        px = jnp.linspace(*self.horizontal_limits, 100)
        py = jnp.linspace(*self.vertical_limits, 100)
        PX, PY = jnp.meshgrid(px, py)
        X = jnp.stack([PX, PY], axis=-1)
        C = jax.vmap(jax.vmap(self._obstacle_cost))(X)
        plt.contourf(PX, PY, C, cmap="Reds", levels=100)

        plt.xlim(*self.horizontal_limits)
        plt.ylim(*self.vertical_limits)
