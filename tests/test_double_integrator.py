import jax.numpy as jnp

from rddp.systems.double_integrator import DoubleIntegrator


def test_dynamics() -> None:
    """Test the basic dynamics of the double integrator."""
    sys = DoubleIntegrator(dt=0.1)
    assert sys.state_shape == (2,)
    assert sys.action_shape == (1,)
    assert sys.observation_shape == (2,)
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0])
    assert jnp.all(sys.f(x, u) == jnp.array([1.2, 2.3]))
    assert jnp.all(sys.g(x) == x)


if __name__ == "__main__":
    test_dynamics()
