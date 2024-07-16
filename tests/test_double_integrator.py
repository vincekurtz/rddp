import jax.numpy as jnp

from rddp.systems.double_integrator import DoubleIntegrator
from rddp.tasks.double_integrator import DoubleIntegratorProblem


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


def test_plot() -> None:
    """Test plotting the double integrator scenario."""
    prob = DoubleIntegratorProblem(num_steps=10)
    prob.plot_scenario()


if __name__ == "__main__":
    test_dynamics()
    test_plot()
