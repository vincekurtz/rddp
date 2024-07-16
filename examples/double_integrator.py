import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.gradient_descent import solve as solve_gd
from rddp.tasks.double_integrator import DoubleIntegratorProblem

# Global planning horizon definition
HORIZON = 10


def solve_with_gradient_descent() -> None:
    """Solve the optimal control problem using simple gradient descent."""
    prob = DoubleIntegratorProblem(num_steps=HORIZON)
    x0 = jnp.array([-1.1, 1.4])
    U, _, _ = solve_gd(prob, x0)

    prob.plot_scenario()
    X = prob.sys.rollout(U, x0)
    plt.plot(X[:, 0], X[:, 1], "o-")
    plt.show()


if __name__ == "__main__":
    solve_with_gradient_descent()
