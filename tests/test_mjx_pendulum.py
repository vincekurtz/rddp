import jax.numpy as jnp
from mujoco import mjx

from rddp.systems.mjx_pendulum import MjxPendulum


def test_pendulum() -> None:
    """Test the basic features of the MJX pendulum dynamics."""
    sys = MjxPendulum()

    assert sys.state_shape == (2,)
    assert sys.action_shape == (1,)
    assert sys.observation_shape == (3,)

    data = sys.make_data()
    assert isinstance(data, mjx.Data)

    data = data.replace(qpos=jnp.array([jnp.pi / 2]), qvel=jnp.array([0.0]))
    u = jnp.array([0.0])

    old_theta, old_theta_dot = data.qpos[0], data.qvel[0]
    data = sys.f(data, u)
    new_theta, new_theta_dot = data.qpos[0], data.qvel[0]

    assert new_theta != old_theta
    assert new_theta_dot != old_theta_dot

    y = sys.g(data)
    y_pred = jnp.array([jnp.cos(new_theta), jnp.sin(new_theta), new_theta_dot])
    assert y.shape == (3,)
    assert jnp.allclose(y, y_pred)


def test_rollout() -> None:
    """Test that we can roll out the pendulum dynamics."""
    sys = MjxPendulum()
    data = sys.make_data()
    data = data.replace(qpos=jnp.array([jnp.pi / 2]), qvel=jnp.array([0.0]))

    control_tape = jnp.zeros((10, 1))
    all_data = sys.rollout(control_tape, data)
    assert all_data.qpos.shape == (10, 1)
    assert all_data.qvel.shape == (10, 1)


def test_simulation() -> None:
    """Test running an interactive simulation of the pendulum."""
    sys = MjxPendulum()
    x0 = sys.make_data()

    policy_fn = lambda y: jnp.array([0.0])

    sys.simulate_and_render(
        x0, policy_fn, num_steps=jnp.inf, fixedcamid=0, cam_type=2
    )


if __name__ == "__main__":
    # test_pendulum()
    # test_rollout()
    test_simulation()
