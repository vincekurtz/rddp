import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax import base
from brax.envs.base import PipelineEnv, State


class BugTrapEnv(PipelineEnv):
    """A robot with unicycle dynamics must escape a U-shaped maze."""

    def __init__(self):
        """Initialize the environment."""
        self.target_position = jnp.array([1.0, 0.0])
        self.horizontal_limits = (-3, 3)
        self.vertical_limits = (-3, 3)
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

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Reset to a fixed initial state."""
        q = jnp.array([-1.0, 0.0, 0.0])  # px, py, theta
        pipeline_state = base.State(
            q=q,
            qd=jnp.zeros(3),
            x=base.Transform.create(pos=jnp.zeros(3)),
            xd=base.Motion.create(vel=jnp.zeros(3)),
            contact=None,
        )
        obs = q
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Forward dynamics, observation, and reward."""
        # Dyanamics
        q = state.pipeline_state.q
        v, w = action
        theta = q[2]
        qdot = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), w])
        q_new = q + 0.1 * qdot
        new_pipeline_state = state.pipeline_state.replace(q=q_new)

        # Observation
        obs = q_new

        # Reward
        action_cost = action.dot(action)
        obstacle_cost = self._obstacle_cost(q_new)
        goal_cost = jnp.linalg.norm(q_new[0:2] - self.target_position) ** 2
        reward = -0.1 * action_cost - 1000 * obstacle_cost - goal_cost

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

        def scan_fn(total_cost: float, obs_pos: jnp.ndarray):
            cost = jnp.exp(-10 * jnp.linalg.norm(x[0:2] - obs_pos) ** 2)
            return total_cost + cost, None

        total_cost, _ = jax.lax.scan(scan_fn, 0.0, self.obs_positions)
        return total_cost

    def plot_scenario(self) -> None:
        """Make a matplotlib visualization of the maze and goal."""
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
