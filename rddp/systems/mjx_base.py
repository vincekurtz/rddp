from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp
import mujoco
from mujoco import mjx

from rddp.systems.base import DynamicalSystem


class MjxDynamicalSystem(DynamicalSystem, ABC):
    """An generic system based on the MuJoCo MJX physics engine."""

    def __init__(self, mjcf_path: str):
        """Initialize the system.

        Note: This class models the state x with an mjx.Data object rather
        than the state vector x = [q, v].

        Args:
            mjcf_path: The path to the MJCF model definition file.
        """
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model = mjx.put_model(mj_model)

    def make_data(self) -> mjx.Data:
        """Create a new data object for storing the system state."""
        return mjx.make_data(self.model)

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of the state space."""
        return (self.model.nq + self.model.nv,)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Shape of the action space."""
        return (self.model.nu,)

    def f(self, x: mjx.Data, u: jnp.ndarray) -> mjx.Data:
        """Forward dynamics function.

        Args:
            x: The current state xₜ.
            u: The control action uₜ to apply.

        Returns:
            The next state xₜ₊₁.
        """
        data = x.replace(ctrl=u)
        return mjx.step(self.model, data)

    @abstractmethod
    def g(self, x: mjx.Data) -> jnp.ndarray:
        """Output/measurment function.

        Args:
            x: The current state.

        Returns:
            The output measurement y.
        """
