from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct
from mujoco import mjx

from rddp import ROOT


@struct.dataclass
class PendulumSwingupConfig:
    """Config dataclass for pendulum swingup."""

    # model path: scene.xml contains ground + other niceties
    model_path: Union[Path, str] = ROOT + "/envs/models/pendulum/scene.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 1

    # the standard deviation of the noise to add to the angular observations
    stdev_obs: float = 0.0

    # Reward function coefficients
    theta_cost_weight: float = 1.0
    theta_dot_cost_weight: float = 0.1
    control_cost_weight: float = 0.001

    # Torque limits
    u_max: float = 1.0

    # Ranges for sampling initial conditions
    qpos_hi: float = jnp.pi
    qpos_lo: float = -jnp.pi
    qvel_hi: float = 8
    qvel_lo: float = -8


class PendulumSwingupEnv(PipelineEnv):
    """Environment for training a torque-constrained pendulum swingup task.

    States: x = (theta, dtheta), shape=(2,)
    Observations: y = (cos(theta)+1, sin(theta), dtheta), shape=(3,)
    Actions: a = tau, the motor torque, shape=(1,)
    """

    def __init__(self, config: Optional[PendulumSwingupConfig] = None) -> None:
        """Initialize the pendulum swingup environtment."""
        if config is None:
            config = PendulumSwingupConfig()
        self.config = config
        mj_model = mujoco.MjModel.from_xml_path(config.model_path)
        sys = mjcf.load_model(mj_model)

        # Initialize the quadratic cost
        self.Q = jnp.diag(
            jnp.array(
                [
                    self.config.theta_cost_weight,
                    self.config.theta_cost_weight,
                    self.config.theta_dot_cost_weight,
                ]
            )
        )
        self.R = jnp.array([[self.config.control_cost_weight]])

        super().__init__(
            sys, n_frames=config.physics_steps_per_control_step, backend="mjx"
        )

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # reset the positions and velocities
        qpos = jax.random.uniform(
            rng1,
            (self.sys.nq,),
            minval=self.config.qpos_lo,
            maxval=self.config.qpos_hi,
        )
        qvel = jax.random.uniform(
            rng2,
            (self.sys.nv,),
            minval=self.config.qvel_lo,
            maxval=self.config.qvel_hi,
        )

        # DEBUG
        qpos *= 0.0
        qvel *= 0.0

        data = self.pipeline_init(qpos, qvel)

        # other state fields
        obs = self._compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {"reward": reward}
        state_info = {"rng": rng, "step": 0}
        return State(data, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment."""
        rng, rng_obs = jax.random.split(state.info["rng"])

        # physics + observation + reward
        action = self.config.u_max * action  # scale so actions are in [-1, 1]
        action = jnp.clip(action, -self.config.u_max, self.config.u_max)
        data = self.pipeline_step(state.pipeline_state, action)  # physics
        obs = self._compute_obs(data, state.info)  # observation
        obs = (
            obs + jax.random.normal(rng_obs, obs.shape) * self.config.stdev_obs
        )  # adding noise to obs
        reward = self._compute_reward(data, state.info)
        done = 0.0  # pendulum just runs for a fixed number of steps

        # updating state
        state.info["step"] = state.info["step"] + 1
        state.info["rng"] = rng
        state.metrics["reward"] = reward
        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
        return state

    def _compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Compute an observation."""
        theta = data.qpos[0]
        # N.B. we use cos(theta) + 1 so the observation is zero at the goal
        obs = jnp.stack((jnp.cos(theta) + 1, jnp.sin(theta), data.qvel[0]))
        return obs

    def _compute_reward(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> jax.Array:
        """Computes the reward for the current environment state.

        Returns:
            reward (shape=(1,)): the reward, maximized at qpos[0] = np.pi.
        """
        y = self._compute_obs(data, info)
        u = data.ctrl

        ly = jnp.einsum("ij,...i,...j->...", self.Q, y, y)
        lu = jnp.einsum("ij,...i,...j->...", self.R, u, u)
        cost = ly + lu

        return -cost

    @property
    def action_size(self) -> int:
        """Size of the action space."""
        return 1

    @property
    def observation_size(self) -> int:
        """Size of the observation space."""
        return 3
