import jax.numpy as jnp

from rddp.tasks.reach_avoid import ReachAvoid


def test_cost() -> None:
    """Test computing the total cost of the reach-avoid problem."""
    prob = ReachAvoid(
        num_steps=10,
        obstacle_position=jnp.array([0.0, 0.0]),
        target_position=jnp.array([1.5, 1.5]),
    )
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


if __name__ == "__main__":
    test_cost()
