##
#
# Dirty example of pendulum swingup with input constraints.
#
# For testing shooting vs direct diffusion methods
#
##

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def f(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Discrete-time forward dynamics of the pendulum.

    Args:
        x: state of the pendulum (angle, angular velocity)
        u: torque applied to the pendulum

    Returns:
        new state of the pendulum
    """
    theta = x[0]
    theta_dot = x[1]
    tau = jnp.tanh(u)[0]  # enforces input limits

    # Mass, length, gravity
    m = 5.0
    g = 9.81
    l = 1.0  # noqa: E741 (ignore ambiguous variable name)

    # Dynamics
    theta_ddot = (tau - m * g * l * jnp.sin(theta - jnp.pi)) / (m * l**2)
    xdot = jnp.array([theta_dot, theta_ddot])

    return x + 0.1 * xdot


def cost(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """The running cost for pendulum swingup.

    Args:
        x: state of the pendulum (angle, angular velocity)
        u: torque applied to the pendulum

    Returns:
        the cost l(x, u)
    """
    # Angle and angular velocity
    theta = x[0]
    theta_dot = x[1]
    return 0.0 * theta**2 + 0.0 * theta_dot**2 + 0.01 * u[0] ** 2


def terminal_cost(x: jnp.ndarray) -> jnp.ndarray:
    """The terminal cost for pendulum swingup.

    Args:
        x: state of the pendulum (angle, angular velocity)

    Returns:
        the terminal cost phi(x)
    """
    theta = x[0]
    theta_dot = x[1]
    return 10.0 * theta**2 + 1.0 * theta_dot**2


def rollout(
    x0: jnp.ndarray, us: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Do a rollout of the pendulum.

    Args:
        x0: initial state of the pendulum
        us: control sequence

    Returns:
        the total cost and state trajectory
    """

    def scan_fn(carry: Tuple, u: jnp.ndarray):
        x, J = carry
        x = f(x, u)
        J += cost(x, u)
        return (x, J), x

    (x, J), X = jax.lax.scan(scan_fn, (x0, 0.0), us)
    J += terminal_cost(x)
    return X, J


def shooting_gradient_descent(
    x0: jnp.ndarray, horizon: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Do simple single-shooting gradient descent.

    Args:
        x0: initial state of the pendulum
        horizon: number of time steps to plan over

    Returns:
        the optimal state trajectory, and control sequence.
    """
    # Parameters
    num_iters = 5000
    print_every = 100
    alpha = 0.1

    # Objective and gradient functions
    def objective(us: jnp.ndarray) -> jnp.ndarray:
        _, J = rollout(x0, us)
        return J

    cost_and_grad = jax.jit(jax.value_and_grad(objective))

    # Optimize
    us = jnp.zeros((horizon, 1))  # initial guess
    for i in range(num_iters):
        J, dJ = cost_and_grad(us)
        us -= alpha * dJ

        if i % print_every == 0 or i == num_iters - 1:
            grad_norm = jnp.linalg.norm(dJ)
            print(f"Iteration {i}, cost {J}, grad norm {grad_norm}")

    return rollout(x0, us)[0], us


def shooting_mppi(
    x0: jnp.ndarray, horizon: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Do simple single-shooting MPPI.

    Args:
        x0: initial state of the pendulum
        horizon: number of time steps to plan over

    Returns:
        the optimal state trajectory
    """
    rng = jax.random.PRNGKey(0)

    # Parameters
    num_iters = 5000
    print_every = 100
    temperature = 0.01
    sigma = 0.1
    num_samples = 32

    # Objective function
    def objective(us: jnp.ndarray) -> jnp.ndarray:
        _, J = rollout(x0, us)
        return J

    jit_objective = jax.jit(objective)
    vmap_objective = jax.jit(jax.vmap(objective))

    # Optimize
    us = jnp.zeros((horizon, 1))  # initial guess
    for i in range(num_iters):
        # Generate noised control tapes
        rng, noise_rng = jax.random.split(rng)
        noise = sigma * jax.random.normal(noise_rng, (num_samples, horizon, 1))
        U = us + noise

        # Evaluate the cost of each noised control tape
        J = vmap_objective(U)
        Jmin = jnp.min(J, axis=0)

        # Take the exponentially weighted average
        w = jnp.exp(-1.0 / temperature * (J - Jmin))
        w /= jnp.sum(w, axis=0)
        us = jnp.einsum("i,ijk->jk", w, U)

        if i % print_every == 0 or i == num_iters - 1:
            J = jit_objective(us)
            print(f"Iteration {i}, cost {J}")

    return rollout(x0, us)[0], us


def make_meshgrid(
    theta_range: Sequence[float],
    theta_dot_range: Sequence[float],
    num_points: int,
) -> jnp.ndarray:
    """Make a meshgrid of states for plotting."""
    th = jnp.linspace(*theta_range, num_points)
    thd = jnp.linspace(*theta_dot_range, num_points)
    TH, THD = jnp.meshgrid(th, thd)
    return jnp.stack([TH, THD], axis=-1)


def plot_vector_field() -> None:
    """Make a plot of the vector field and cost contours."""
    # Set up the plot
    theta_range = (-1.5 * jnp.pi, 1.5 * jnp.pi)
    theta_dot_range = (-8.0, 8.0)
    plt.xlim(*theta_range)
    plt.ylim(*theta_dot_range)

    # Vector field
    X = make_meshgrid(theta_range, theta_dot_range, 20)
    U = jnp.zeros((20, 20, 1))
    dX = jax.vmap(jax.vmap(f))(X, U) - X
    plt.quiver(X[:, :, 0], X[:, :, 1], dX[:, :, 0], dX[:, :, 1], color="k")
    plt.xlabel("Angle (rad)")
    plt.ylabel("Angular velocity (rad/s)")

    # Cost contours
    X = make_meshgrid(theta_range, theta_dot_range, 100)
    C = jax.vmap(jax.vmap(terminal_cost))(X)
    plt.contourf(X[:, :, 0], X[:, :, 1], C, levels=20, alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label("Cost")


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1)

    # X, U = shooting_gradient_descent(jnp.array([3.0, -1.0]), 50)
    X, U = shooting_mppi(jnp.array([3.0, 1.0]), 50)

    # Plot the state trajectory
    plt.sca(ax[0])
    plot_vector_field()
    plt.plot(X[:, 0], X[:, 1], "o-")

    # Plot the control tape over time
    plt.sca(ax[1])
    plt.plot(jnp.tanh(U))
    plt.xlabel("Time step")
    plt.ylabel("Control torque")
    plt.ylim(-1.1, 1.1)

    plt.show()
