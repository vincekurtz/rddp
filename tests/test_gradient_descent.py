import jax.numpy as jnp

from rddp.gradient_descent import solve
from rddp.tasks.reach_avoid import ReachAvoid


def test_gradient_descent() -> None:
    """Test solving the reach-avoid problem using gradient descent."""
    prob = ReachAvoid(num_steps=10)
    x0 = jnp.array([0.1, -1.5])

    U, J, grad = solve(prob, x0, max_iter=10_000)
    X = prob.sys.rollout(U, x0)
    final_err = jnp.linalg.norm(X[-1, :] - prob.config.target_position)

    assert jnp.linalg.norm(grad) < 1e-3
    assert J < 0.2
    assert final_err < 1e-1


if __name__ == "__main__":
    test_gradient_descent()
