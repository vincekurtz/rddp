import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax import base
from brax.envs.base import PipelineEnv, State


class PendulumEnv(PipelineEnv):
    """An inverted pendulum must swing up and balance."""

    def __init__(self):
        """Initialize the pendulum environment."""
        self.mass = 1.0
        self.length = 1.0
        self.gravity = 9.81

        self.position_limits = (-1.5 * jnp.pi, 1.5 * jnp.pi)
        self.velocity_limits = (-10.0, 10.0)

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Reset to a new initial state.

        Args:
            rng: random seed

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
        obs = jnp.array([jnp.cos(pos), jnp.sin(pos), vel])
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
            action: the action (torque) to take.

        Return:
            the next state, including the updated reward and observation.
        """
        # Forward dynamics
        theta = state.pipeline_state.q[0]
        theta_dot = state.pipeline_state.qd[0]
        tau = action[0]
        mgl = self.mass * self.gravity * self.length
        ml2 = self.mass * self.length**2
        theta_ddot = (tau - mgl * jnp.sin(theta - jnp.pi)) / ml2
        new_theta = theta + 0.1 * theta_dot
        new_theta_dot = theta_dot + 0.1 * theta_ddot
        new_pipeline_state = state.pipeline_state.replace(
            q=jnp.array([new_theta]),
            qd=jnp.array([new_theta_dot]),
        )

        # Observation
        obs = jnp.array([jnp.cos(new_theta), jnp.sin(new_theta), new_theta_dot])

        # Reward
        input_cost = tau**2
        theta_error = jnp.sin(new_theta) ** 2 + (jnp.cos(new_theta) - 1.0) ** 2
        theta_cost = theta_error
        theta_dot_cost = new_theta_dot**2

        reward = -(0.1 * theta_cost + 0.1 * theta_dot_cost + 0.001 * input_cost)

        return state.replace(
            pipeline_state=new_pipeline_state,
            obs=obs,
            reward=reward,
        )

    @property
    def observation_size(self) -> int:
        """Returns the size of the observation."""
        return 3

    @property
    def action_size(self) -> int:
        """Returns the size of the action."""
        return 1

    def plot_scenario(self) -> None:
        """Plot the vector field and target."""
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
        plt.xlabel("theta")
        plt.ylabel("theta_dot")
