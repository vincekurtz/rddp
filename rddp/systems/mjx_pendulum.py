from typing import Tuple

import jax.numpy as jnp
from mujoco import mjx

from rddp.systems.mjx_base import MjxDynamicalSystem


class MjxPendulum(MjxDynamicalSystem):
    """A mujoco MJX model of an inverted pendulum."""

    def __init__(self) -> None:
        """Initialize the pendulum system."""
        super().__init__(
            "rddp/systems/models/pendulum_scene.xml",
            sim_steps_per_control_step=2,
        )

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of the observation."""
        return (3,)

    def g(self, x: mjx.Data) -> jnp.ndarray:
        """Observation function."""
        theta = x.qpos[0]
        theta_dot = x.qvel[0]
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])
