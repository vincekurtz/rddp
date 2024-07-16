import time
from typing import Tuple

import jax
import jax.numpy as jnp

from rddp.tasks.base import OptimalControlProblem


def solve(
    prob: OptimalControlProblem,
    x0: jnp.ndarray,
    u_guess: jnp.ndarray = None,
    max_iter: int = 5000,
    step_size: float = 0.01,
    print_every: int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve a trajectory optimization using vanilla gradient descent.

    Args:
        prob: The optimal control problem.
        x0: The initial state.
        u_guess: An initial guess for the control trajectory.
        max_iter: The maximum number of iterations.
        step_size: The gradient descent step size.
        print_every: The frequency of printing the cost.

    Returns:
        The optimal control tape, cost, and gradient at the last iteration.
    """
    # Initialize the control tape.
    if u_guess is None:
        U = jnp.zeros((prob.num_steps, *prob.sys.action_shape))
    else:
        U = u_guess

    # Define cost and gradient functions.
    cost_and_grad = jax.jit(
        jax.value_and_grad(lambda us: prob.total_cost(us, x0))
    )

    # Run gradient descent.
    st = time.time()
    for i in range(max_iter):
        J, grad = cost_and_grad(U)
        U -= step_size * grad
        if i % print_every == 0:
            print(f"Iter {i}, cost {J:.4f}, grad {jnp.linalg.norm(grad):.4f}")
    print(f"Gradient descent took {time.time() - st:.2f} seconds")

    return U, J, grad
