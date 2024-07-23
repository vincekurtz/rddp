##
#
# Test different dataset generation ideas with the bug trap problem.
#
##

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.tasks.bug_trap import BugTrap
from rddp.utils import DiffusionDataset, sample_dataset


def get_gradient(prob: BugTrap, u: jnp.ndarray, x0: jnp.ndarray) -> jnp.ndarray:
    """Get the (analytical) gradient of the cost function."""
    J = lambda u: prob.total_cost(u, x0)
    grad_J = jax.grad(J)
    return grad_J(u)


def visualize_dataset(
    dataset: DiffusionDataset, prob: BugTrap, num_noise_levels: int
) -> None:
    """Make some plots of the generated dataset.

    Args:
        dataset: The generated dataset.
        prob: The reach-avoid problem instance to use for plotting.
        num_noise_levels: The number of noise levels in the dataset.
    """
    rng = jax.random.PRNGKey(0)

    noise_levels = [
        0,
        int(num_noise_levels / 4),
        int(num_noise_levels / 2),
        int(3 * num_noise_levels / 4),
        num_noise_levels - 1,
    ]

    # Plot samples at certain iterations
    fig, ax = plt.subplots(1, len(noise_levels))
    for i, k in enumerate(noise_levels):
        plt.sca(ax[i])

        # Get a random subset of the data at this noise level
        rng, sample_rng = jax.random.split(rng)
        subset = sample_dataset(dataset, k, 32, sample_rng)

        # Plot the scenario and the sampled trajectories
        prob.plot_scenario()
        Xs = jax.vmap(prob.sys.rollout)(subset.U, subset.y0)  # N.B. y = x
        px, py = Xs[:, :, 0].T, Xs[:, :, 1].T
        ax[i].plot(px, py, "o-", color="blue", alpha=0.5)

        sigma = subset.sigma[0, 0]
        ax[i].set_title(f"k={k}, σₖ={sigma:.4f}")

    # Plot costs at each iteration
    plt.figure()
    jit_cost = jax.jit(jax.vmap(prob.total_cost))
    for k in range(num_noise_levels):
        iter = num_noise_levels - k

        # Get a random subset of the data at this noise level
        rng, sample_rng = jax.random.split(rng)
        subset = sample_dataset(dataset, k, 32, sample_rng)

        # Compute the cost of each trajectory and add it to the plot
        # N.B. for this example we have y = x.
        costs = jit_cost(subset.U, subset.y0)
        plt.scatter(jnp.ones_like(costs) * iter, costs, color="blue", alpha=0.5)
    plt.xlabel("Iteration (L - k)")
    plt.ylabel("Cost J(U, x₀)")
    plt.yscale("log")

    plt.show()


def gradient_descent() -> DiffusionDataset:
    """Generate with simple gradient descent from random initial conditions."""
    rng = jax.random.PRNGKey(0)

    # Parameters
    horizon = 20
    num_samples = 128
    num_iterations = 5000
    starting_noise_level = 0.5
    learning_rate = 0.001

    # Problem setup
    prob = BugTrap(num_steps=horizon)
    x0 = prob.sample_initial_state(rng)
    grad = lambda u: get_gradient(prob, u, x0)
    vmap_grad = jax.vmap(grad)

    # Sample initial control sequences from a normal distribution
    rng, noise_rng = jax.random.split(rng)
    noise_shape = (num_samples, horizon, *prob.sys.action_shape)
    U = starting_noise_level * jax.random.normal(noise_rng, shape=noise_shape)

    def scan_fn(control_tape: jnp.ndarray, k: int):
        s = vmap_grad(control_tape)
        U = control_tape - learning_rate * s
        dataset = DiffusionDataset(
            y0=jnp.tile(x0, (num_samples, 1)),
            U=U,
            s=s,
            k=jnp.tile(jnp.array([k]), (num_samples, 1)),
            sigma=jnp.tile(jnp.array([jnp.nan]), (num_samples, 1)),
        )
        return U, dataset

    print("Generating dataset...")
    U, dataset = jax.lax.scan(scan_fn, U, jnp.arange(num_iterations, -1, -1))

    # Flatten the dataset
    print("Flattening dataset...")
    dataset = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), dataset)

    # Visualize the dataset
    print("Visualizing dataset...")
    visualize_dataset(dataset, prob, num_iterations)


if __name__ == "__main__":
    dataset = gradient_descent()
