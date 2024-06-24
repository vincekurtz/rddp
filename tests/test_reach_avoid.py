import jax
import jax.numpy as jnp

from rddp.tasks.reach_avoid import ReachAvoid


def test_cost() -> None:
    """Test computing the total cost of the reach-avoid problem."""
    prob = ReachAvoid(num_steps=10)
    x0 = jnp.array([-1.0, -1.0])
    control_tape = [
        jnp.array([jnp.sin(t), jnp.cos(0.1 * t)]) for t in range(10)
    ]
    control_tape = jnp.array(control_tape)

    # Manually compute the total cost.
    cost = 0.0
    x = x0
    for t in range(10):
        u = control_tape[t]
        cost += prob.running_cost(x, u, t)
        x = x + u
    cost += prob.terminal_cost(x)

    # Compute the total cost using the problem's method.
    cost2 = prob.total_cost(control_tape, x0)
    assert jnp.isclose(cost, cost2)


def test_plot() -> None:
    """Test plotting the reach-avoid scenario."""
    prob = ReachAvoid(num_steps=10)
    prob.plot_scenario()


def test_sample_initial_state() -> None:
    """Test sampling the initial state of the reach-avoid problem."""
    rng = jax.random.PRNGKey(0)
    prob = ReachAvoid(num_steps=10)
    x0 = prob.sample_initial_state(rng)
    assert x0.shape == (2,)


if __name__ == "__main__":
    test_cost()
    test_sample_initial_state()
    test_plot()
