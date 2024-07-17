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


if __name__ == "__main__":
    test_pendulum()
