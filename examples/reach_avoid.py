import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.data_generation import (
    AnnealedLangevinOptions,
    DatasetGenerationOptions,
    DatasetGenerator,
)
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
    U = jnp.zeros((prob.num_steps - 1, prob.sys.action_shape[0]))
    J, grad = cost_and_grad(U)

    for i in range(5000):
        J, grad = cost_and_grad(U)
        U -= 1e-2 * grad

        if i % 1000 == 0:
            print(f"Step {i}, cost {J}, grad {jnp.linalg.norm(grad)}")

    return prob.sys.rollout(U, x0)


def plot_dataset() -> None:
    """Generate some data and make plots of it."""
    rng = jax.random.PRNGKey(0)
    x0 = jnp.array([0.1, -1.5])

    # Problem setup
    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        temperature=0.01,
        num_noise_levels=100,
        starting_noise_level=0.1,
        noise_decay_rate=0.95,
    )
    gen_options = DatasetGenerationOptions(
        num_initial_states=1,
        num_data_points_per_initial_state=16,
        num_rollouts_per_data_point=32,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate some data
    rng, gen_rng = jax.random.split(rng)
    (x0s, Us, scores, ks) = generator.generate_from_state(x0, gen_rng)

    # Make some plots
    gamma = langevin_options.noise_decay_rate
    sigma_L = langevin_options.starting_noise_level

    # Plot samples at certain noise levels
    fig, ax = plt.subplots(1, 4)

    for i, k in enumerate([0, 25, 50, 75]):
        plt.sca(ax[i])
        prob.plot_scenario()
        ax[i].set_title(f"Noise level {k}")
        Xs = jax.vmap(prob.sys.rollout)(Us[k], x0s[k])
        px = Xs[:, :, 0].T
        py = Xs[:, :, 1].T
        ax[i].plot(px, py, "o-", color="blue", alpha=0.5)

    # Plot costs over iterations
    fig, ax = plt.subplots(2, 1, sharex=True)
    iters = langevin_options.num_noise_levels - ks
    costs = jax.vmap(jax.vmap(prob.total_cost))(Us, x0s)
    noise_levels = sigma_L * gamma**iters

    ax[0].scatter(iters, costs, c="blue", alpha=0.5)
    ax[0].set_ylabel("Cost")
    ax[1].scatter(iters, noise_levels, c="red", alpha=0.5)
    ax[1].set_ylabel("Noise level")
    ax[1].set_xlabel("Iteration")

    print("Averge final cost:", jnp.mean(costs[-1]))

    plt.show()


if __name__ == "__main__":
    plot_dataset()

    # prob = ReachAvoid(num_steps=20)
    # prob.plot_scenario()
    # x0 = jnp.array([0.1, -1.5])
    # X = solve_with_gradient_descent(prob, x0)
    # plt.plot(X[:, 0], X[:, 1], "o-")
    # plt.show()
