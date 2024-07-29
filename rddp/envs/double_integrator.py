import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax import base
from brax.envs.base import PipelineEnv, State


class DoubleIntegratorEnv(PipelineEnv):
    """A simple 1D double integrator must go to the origin."""

    def __init__(self):
        """Initialize the double integrator environment."""
        self.position_limits = (-3, 3)
        self.velocity_limits = (-3, 3)

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Reset the environment to a new initial state.

        Args:
            rng: The random seed

        Returns:
            The initial state of the environment.
        """
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        pos = jax.random.uniform(
            pos_rng,
            minval=self.position_limits[0],
            maxval=self.position_limits[1],
        )
        vel = jax.random.uniform(
            vel_rng,
            minval=self.velocity_limits[0],
            maxval=self.velocity_limits[1],
        )
        pipeline_state = base.State(
            q=jnp.array([pos]),
            qd=jnp.array([vel]),
            x=base.Transform.create(pos=jnp.zeros(3)),
            xd=base.Motion.create(vel=jnp.zeros(3)),
            contact=None,
        )
        obs = jnp.array([pos, vel])
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=0.0,
            done=0.0,
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Forward dynamics step.

        Args:
            state: container for the system state and reward.
            action: the action (acceleration) to take.

        Return:
            the next state, including the updated reward and observation.
        """
        # Forward dynamics
        pos = state.pipeline_state.q
        vel = state.pipeline_state.qd
        new_pos = pos + 0.1 * vel  # dt = 0.1
        new_vel = vel + 0.1 * action
        new_pipeline_state = state.pipeline_state.replace(q=new_pos, qd=new_vel)

        # Observation
        obs = jnp.array([new_pos[0], new_vel[0]])

        # Reward
        action_cost = 0.01 * action.dot(action)
        velocity_cost = 0.1 * vel.dot(vel)
        position_cost = pos.dot(pos)
        reward = -(action_cost + position_cost + velocity_cost)

        return state.replace(
            pipeline_state=new_pipeline_state,
            obs=obs,
            reward=reward,
        )

    @property
    def action_size(self) -> int:
        """Actions are accelerations."""
        return 1

    @property
    def observation_size(self) -> int:
        """Observations are positions and velocities."""
        return 2

    def plot_scenario(self) -> None:
        """Plot the vector field and target."""
        # Green star at the target state
        plt.plot(0.0, 0.0, "g*", markersize=20)

        # Sample some initial states in a grid
        p = jnp.linspace(*self.position_limits, 20)
        v = jnp.linspace(*self.velocity_limits, 20)
        P, V = jnp.meshgrid(p, v)

        # Forward dynamics from these initial states
        rng = jax.random.split(jax.random.PRNGKey(0), (20, 20))
        X = jax.vmap(jax.vmap(self.reset))(rng)
        X = X.tree_replace(
            {"pipeline_state.q": P[None].T, "pipeline_state.qd": V[None].T}
        )
        X_next = jax.vmap(jax.vmap(lambda x: self.step(x, jnp.zeros(1))))(X)

        # Vector field plot
        P = X.pipeline_state.q
        V = X.pipeline_state.qd
        dP = X_next.pipeline_state.q - P
        dV = X_next.pipeline_state.qd - V
        plt.quiver(P[..., 0], V[..., 0], dP[..., 0], dV[..., 0], color="k")

        plt.xlim(self.position_limits)
        plt.ylim(self.velocity_limits)
        plt.xlabel("Position")
        plt.ylabel("Velocity")
