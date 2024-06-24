import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.tasks.base import OptimalControlProblem
from rddp.tasks.reach_avoid import ReachAvoid


def solve_with_gradient_descent(
    prob: OptimalControlProblem, x0: jnp.ndarray
) -> jnp.ndarray:
    """Solve the optimal control problem using simple gradient descent.

    Args:
        prob: The optimal control problem.
        x0: The initial state.

    Returns:
        The optimal state trajectory.
    """
    cost_and_grad = jax.jit(
        jax.value_and_grad(lambda us: prob.total_cost(us, x0))
    )
    U = jnp.zeros((prob.num_steps, prob.sys.action_shape[0]))
    J, grad = cost_and_grad(U)

    for i in range(5000):
        J, grad = cost_and_grad(U)
        U -= 1e-2 * grad

        if i % 1000 == 0:
            print(f"Step {i}, cost {J}, grad {jnp.linalg.norm(grad)}")

    return prob.sys.rollout(U, x0)


if __name__ == "__main__":
    prob = ReachAvoid(num_steps=20)
    prob.plot_scenario()

    x0 = jnp.array([0.1, -1.5])
    X = solve_with_gradient_descent(prob, x0)
    plt.plot(X[:, 0], X[:, 1], "o-")

    plt.show()
