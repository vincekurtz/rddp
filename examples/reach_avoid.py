import pickle
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rddp.architectures import DiffusionPolicyMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.tasks.reach_avoid import ReachAvoid
from rddp.training import TrainingOptions, train
from rddp.utils import (
    AnnealedLangevinOptions,
    DiffusionDataset,
    HDF5DiffusionDataset,
    annealed_langevin_sample,
    sample_dataset,
)

# Global planning horizon definition
HORIZON = 40


class ReachAvoidFixedX0(ReachAvoid):
    """A reach-avoid problem with a fixed initial state."""

    def __init__(self, num_steps: int, start_state: jnp.ndarray):
        """Initialize the reach-avoid problem.

        Args:
            num_steps: The number of time steps T.
            start_state: The initial state x0.
        """
        super().__init__(num_steps)
        self.x0 = start_state

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample the initial state x₀."""
        return self.x0


def solve_with_gradient_descent(
    plot: bool = True, u_guess: float = 0.0
) -> None:
    """Solve the optimal control problem using simple gradient descent."""
    prob = ReachAvoid(num_steps=HORIZON)
    x0 = jnp.array([0.1, -1.5])

    cost_and_grad = jax.jit(
        jax.value_and_grad(lambda us: prob.total_cost(us, x0))
    )
    U = u_guess * jnp.ones((prob.num_steps - 1, prob.sys.action_shape[0]))
    J, grad = cost_and_grad(U)

    st = time.time()
    for i in range(5000):
        J, grad = cost_and_grad(U)
        U -= 1e-2 * grad

        if i % 1000 == 0:
            print(f"Step {i}, cost {J}, grad {jnp.linalg.norm(grad)}")
    print(f"Gradient descent took {time.time() - st:.2f} seconds")

    if plot:
        X = prob.sys.rollout(U, x0)
        prob.plot_scenario()
        plt.plot(X[:, 0], X[:, 1], "o-")
        plt.show()
    return U


def visualize_dataset(
    dataset: DiffusionDataset, prob: ReachAvoid, num_noise_levels: int
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
        Xs = jax.vmap(prob.sys.rollout)(subset.U, subset.x0)
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
        costs = jit_cost(subset.U, subset.x0)
        plt.scatter(jnp.ones_like(costs) * iter, costs, color="blue", alpha=0.5)
    plt.xlabel("Iteration (L - k)")
    plt.ylabel("Cost J(U, x₀)")
    plt.yscale("log")

    plt.show()


def generate_dataset(plot: bool = False) -> None:
    """Generate some data and make plots of it."""
    rng = jax.random.PRNGKey(0)
    x0 = jnp.array([-0.1, -1.5])
    save_path = "/tmp/reach_avoid/"

    # Problem setup
    prob = ReachAvoidFixedX0(num_steps=HORIZON, start_state=x0)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=500,
        starting_noise_level=0.5,
        num_steps=100,
        step_size=0.01,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        temperature=0.001,
        num_initial_states=256,
        num_rollouts_per_data_point=128,
        save_every=100,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate some data
    st = time.time()
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng)
    print(f"Data generation took {time.time() - st:.2f} seconds")

    # Make some plots if requested
    if plot:
        # Select every Nth data point for visualization. This avoids loading
        # the full dataset into memory.
        N = 10
        st = time.time()
        with h5py.File(save_path + "dataset.h5", "r") as f:
            h5_dataset = HDF5DiffusionDataset(f)
            idxs = jnp.arange(0, len(h5_dataset), N)
            print("Loading...")
            dataset = h5_dataset[idxs]
        print(f"Loaded dataset in {time.time() - st:.2f} seconds")
        visualize_dataset(dataset, prob, langevin_options.num_noise_levels)


def fit_score_model() -> None:
    """Fit a simple score model to the generated data."""
    # Specify location of the training data
    data_dir = "/tmp/reach_avoid/"

    # Load the langiven sampling options
    with open(data_dir + "langevin_options.pkl", "rb") as f:
        langevin_options = pickle.load(f)

    # Set up the training options and the score network
    training_options = TrainingOptions(
        batch_size=5120,
        num_superbatches=20,
        epochs=50,
        learning_rate=1e-3,
    )
    net = DiffusionPolicyMLP(layer_sizes=(512,) * 3)

    # Train the score network
    st = time.time()
    params, metrics = train(net, data_dir + "dataset.h5", training_options)
    print(f"Training took {time.time() - st:.2f} seconds")

    # Save the trained model and parameters
    fname = "/tmp/reach_avoid_score_model.pkl"
    with open(fname, "wb") as f:
        data = {
            "params": params,
            "net": net,
            "langevin_options": langevin_options,
        }
        pickle.dump(data, f)
    print(f"Saved trained model to {fname}")


def deploy_trained_model(
    plot: bool = True, animate: bool = False, save_path: str = None
) -> None:
    """Use the trained model to generate optimal actions."""
    rng = jax.random.PRNGKey(0)

    # Set up the system
    x0 = jnp.array([-0.1, -1.5])
    prob = ReachAvoidFixedX0(num_steps=HORIZON, start_state=x0)

    # Load the trained score network
    with open("/tmp/reach_avoid_score_model.pkl", "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    net = data["net"]
    options = data["langevin_options"]

    # Decide how much noise to add in the Langevin sampling
    options = options.replace(noise_injection_level=0.0)

    def optimize_control_tape(rng: jax.random.PRNGKey):
        """Optimize the control sequence using Langevin dynamics."""
        # Guess an initial control sequence
        rng, guess_rng = jax.random.split(rng, 2)
        U_guess = options.starting_noise_level * jax.random.normal(
            guess_rng, (prob.num_steps - 1, 2)
        )

        # Do annealed langevin sampling
        rng, langevin_rng = jax.random.split(rng)
        U, data = annealed_langevin_sample(
            options=options,
            x0=x0,
            u_init=U_guess,
            score_fn=lambda x, u, sigma, rng: net.apply(
                params, x, u, jnp.array([sigma])
            ),
            rng=langevin_rng,
        )

        return U, data

    # Optimize from a bunch of initial guesses
    num_samples = 32
    rng, opt_rng = jax.random.split(rng)
    opt_rng = jax.random.split(opt_rng, num_samples)
    st = time.time()
    Us, data = jax.vmap(optimize_control_tape)(opt_rng)
    print(f"Sample generation took {time.time() - st:.2f} seconds")
    Xs = jax.vmap(prob.sys.rollout, in_axes=(0, None))(Us, x0)
    costs = jax.vmap(prob.total_cost, in_axes=(0, None))(Us, x0)
    print(f"Cost: {jnp.mean(costs):.4f} +/- {jnp.std(costs):.4f}")

    # Plot the sampled trajectories
    if plot:
        plt.figure()
        prob.plot_scenario()
        for i in range(num_samples):
            plt.plot(Xs[i, :, 0], Xs[i, :, 1], "o-", color="blue", alpha=0.5)
        plt.show()

    # Animate the trajectory generation process
    if animate:
        x0 = data.x0[:, :, -1, :]  # take the last sample at each noise level
        U = data.U[:, :, -1, :]
        sigma = data.sigma[:, :, -1]
        Xs = jax.vmap(jax.vmap(prob.sys.rollout))(U, x0)

        fig, ax = plt.subplots()
        prob.plot_scenario()

        paths = []
        for _ in range(num_samples):
            paths.append(ax.plot([], [], "o-")[0])

        def update(i: int):
            ax.set_title(f"σₖ={sigma[0, i, 0]:.4f}")
            for j, path in enumerate(paths):
                path.set_data(Xs[j, i, :, 0], Xs[j, i, :, 1])
            return path

        anim = FuncAnimation(
            fig, update, frames=options.num_noise_levels, interval=10
        )
        if save_path is not None:
            anim.save(save_path, writer="ffmpeg", fps=60)
            print(f"Saved animation to {save_path}")
        plt.show()


if __name__ == "__main__":
    usage = "Usage: python reach_avoid.py [generate|fit|deploy|gd|animate]"

    num_args = 1
    if len(sys.argv) != num_args + 1:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == "generate":
        generate_dataset(plot=True)
    elif sys.argv[1] == "fit":
        fit_score_model()
    elif sys.argv[1] == "deploy":
        deploy_trained_model()
    elif sys.argv[1] == "gd":
        solve_with_gradient_descent()
    elif sys.argv[1] == "animate":
        deploy_trained_model(plot=False, animate=True)
    else:
        print(usage)
        sys.exit(1)
