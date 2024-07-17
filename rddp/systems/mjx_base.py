import time
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from rddp.systems.base import DynamicalSystem


class MjxDynamicalSystem(DynamicalSystem, ABC):
    """An generic system based on the MuJoCo MJX physics engine."""

    def __init__(self, mjcf_path: str, sim_steps_per_control_step: int = 1):
        """Initialize the system.

        Note: This class models the state x with an mjx.Data object rather
        than the state vector x = [q, v].

        Args:
            mjcf_path: The path to the MJCF model definition file.
            sim_steps_per_control_step: The number of simulation steps to take
                per control step. This can be used to simulate at a higher
                frequency than the control frequency.
        """
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model = mjx.put_model(self.mj_model)
        self.sim_steps_per_control_step = sim_steps_per_control_step

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
        x_next, _ = jax.lax.scan(
            lambda state, _: (mjx.step(self.model, state), None),
            x.replace(ctrl=u),
            jnp.arange(self.sim_steps_per_control_step),
        )
        return x_next

    @abstractmethod
    def g(self, x: mjx.Data) -> jnp.ndarray:
        """Output/measurment function.

        Args:
            x: The current state.

        Returns:
            The output measurement y.
        """

    def simulate_and_render(
        self,
        x0: mjx.Data,
        policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
        num_steps: int,
        fixedcamid: int = -1,
        cam_type: int = 0,
    ) -> None:
        """Run an interactive mujoco simulation with the given policy.

        Args:
            x0: The initial system state.
            policy_fn: The policy function to apply, u = policy_fn(y).
            num_steps: The number of steps to simulate.
            fixedcamid: The camera ID to use for visualization.
            cam_type: The camera type to use for visualization.
        """
        mj_data = mujoco.MjData(self.mj_model)
        mj_data.qpos[:] = x0.qpos
        mj_data.qvel[:] = x0.qvel

        steps = 0
        dt = self.mj_model.opt.timestep * self.sim_steps_per_control_step
        with mujoco.viewer.launch_passive(self.mj_model, mj_data) as viewer:
            viewer.cam.fixedcamid = fixedcamid
            viewer.cam.type = cam_type

            while viewer.is_running() and steps < num_steps:
                start_time = time.time()

                # Get the observation
                mjx_data = mjx.put_data(self.mj_model, mj_data)
                y = self.g(mjx_data)

                # Get an action from the policy
                u = policy_fn(y)

                # Apply the policy and step the simulation
                mj_data.ctrl[:] = np.array(u)
                for _ in range(self.sim_steps_per_control_step):
                    mujoco.mj_step(self.mj_model, mj_data)
                    viewer.sync()

                # Try to run in roughly realtime
                elapsed_time = time.time() - start_time
                if elapsed_time < dt:
                    time.sleep(dt - elapsed_time)

                steps += 1
