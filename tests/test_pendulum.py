import jax.numpy as jnp

from rddp.systems.pendulum import Pendulum
from rddp.tasks.pendulum_swingup import PendulumSwingupProblem


def test_dynamics() -> None:
    """Test the basic dynamics of the pendulum."""
    sys = Pendulum(dt=0.1)

    # At the upright position, the pendulum should not move.
    x = jnp.array([0.0, 0.0])
    u = jnp.array([0.0])
    x_next = sys.f(x, u)
    assert jnp.allclose(x_next, jnp.array([0.0, 0.0]), atol=1e-6)

    # At the upright position, the observation should be [1, 0, 0].
    y = sys.g(x)
    assert jnp.allclose(y, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_plot() -> None:
    """Test the plot_scenario function."""
    prob = PendulumSwingupProblem(num_steps=10)
    prob.plot_scenario()


if __name__ == "__main__":
    test_dynamics()
    test_plot()
