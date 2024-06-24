import jax.numpy as jnp

from rddp.systems.single_integrator import SingleIntegrator


def test_dynamics() -> None:
    """Test the basic dynamics of the single integrator."""
    sys = SingleIntegrator(dt=0.1)
    assert sys.state_shape == (2,)
    assert sys.action_shape == (2,)
    assert sys.observation_shape == (2,)
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0, 4.0])
    assert jnp.all(sys.f(x, u) == jnp.array([1.3, 2.4]))
    assert jnp.all(sys.g(x) == x)


def test_simulation() -> None:
    """Test the simulation of the single integrator."""
    sys = SingleIntegrator(dt=0.1)
    x0 = jnp.array([0.0, 0.0])
    policy_fn = lambda x: jnp.array([1.0, 1.0])
    sys.simulate_and_render(x0, policy_fn, num_steps=10)


if __name__ == "__main__":
    test_dynamics()
    test_simulation()
