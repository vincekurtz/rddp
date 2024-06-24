import jax.numpy as jnp

from rddp.systems.single_integrator import SingleIntegrator


def test_dynamics() -> None:
    """Test the basic dynamics of the single integrator."""
    system = SingleIntegrator(dt=0.1)
    assert system.state_shape == (2,)
    assert system.action_shape == (2,)
    assert system.observation_shape == (2,)
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0, 4.0])
    assert jnp.all(system.f(x, u) == jnp.array([1.3, 2.4]))
    assert jnp.all(system.g(x) == x)


if __name__ == "__main__":
    test_dynamics()
