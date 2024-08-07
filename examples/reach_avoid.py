import pickle
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rddp.architectures import ScoreMLP
from rddp.envs.reach_avoid import ReachAvoidEnv
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.gradient_descent import solve as solve_gd
from rddp.ocp import OptimalControlProblem
from rddp.training import TrainingOptions, train
from rddp.utils import (
    AnnealedLangevinOptions,
    DiffusionDataset,
    HDF5DiffusionDataset,
    annealed_langevin_sample,
    sample_dataset,
)

# Global planning horizon definition
HORIZON = 10


def solve_with_gradient_descent(
    plot: bool = True, u_guess: float = 0.0
) -> None:
    """Solve the optimal control problem using simple gradient descent."""
    prob = OptimalControlProblem(
        ReachAvoidEnv(num_steps=HORIZON), num_steps=HORIZON
    )
    x0 = prob.env.reset(jax.random.PRNGKey(0))
    x0 = x0.tree_replace({"pipeline_state.q": jnp.array([0.1, -1.5])})
    u_guess = u_guess * jnp.ones((prob.num_steps - 1, prob.env.action_size))

    U, _, _ = solve_gd(prob, x0, u_guess)

    if plot:
        _, states = prob.rollout(x0, U)
        positions = states.pipeline_state.q
        prob.env.plot_scenario()
        plt.plot(positions[:, 0], positions[:, 1], "o-")
        plt.show()
    return U


def visualize_dataset(
    dataset: DiffusionDataset,
    prob: OptimalControlProblem,
    num_noise_levels: int,
) -> None:
    """Make some plots of the generated dataset.

    Args:
        dataset: The generated dataset.
        prob: The problem instance to use for plotting.
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
        obs = subset.Y

        # Plot the scenario and the sampled trajectories
        prob.env.plot_scenario()
        px, py = obs[..., 0].T, obs[..., 1].T
        ax[i].plot(px, py, "o-", color="blue", alpha=0.5)

        sigma = subset.sigma[0, 0]
        ax[i].set_title(f"k={k}, σₖ={sigma:.4f}")

    # Plot costs across iterations
    plt.figure()
    for k in range(0, num_noise_levels, 50):
        iter = num_noise_levels - k

        # Get a random subset of the data at this noise level
        rng, sample_rng = jax.random.split(rng)
        subset = sample_dataset(dataset, k, 32, sample_rng)

        # Compute the cost of each trajectory and add it to the plot
        costs = subset.cost
        plt.scatter(jnp.ones_like(costs) * iter, costs, color="blue", alpha=0.5)
    plt.xlabel("Iteration (L - k)")
    plt.ylabel("Cost J(U, x₀)")
    plt.yscale("log")

    plt.show()


def generate_dataset(plot: bool = False) -> None:
    """Generate some data and make plots of it."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/reach_avoid/"

    # Problem setup
    prob = OptimalControlProblem(
        ReachAvoidEnv(num_steps=HORIZON),
        num_steps=HORIZON,
    )
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=2000,
        starting_noise_level=0.1,
        step_size=0.03,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=256,
        num_rollouts_per_data_point=128,
        save_every=1000,
        print_every=500,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate some data
    st = time.time()
    rng, gen_rng = jax.random.split(rng)
    generator.generate(gen_rng)
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
        num_superbatches=1,
        epochs=50,
        learning_rate=1e-3,
    )
    net = ScoreMLP(layer_sizes=(128,) * 3)

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


def deploy_trained_model(plot: bool = True, animate: bool = False) -> None:
    """Use the trained model to generate optimal actions."""
    rng = jax.random.PRNGKey(0)
    prob = OptimalControlProblem(
        ReachAvoidEnv(num_steps=HORIZON),
        num_steps=HORIZON,
    )

    def rollout_from_obs(y0: jnp.ndarray, u: jnp.ndarray):
        """Do a rollout from an observation, and return observations."""
        x0 = prob.env.reset(rng)
        x0 = x0.tree_replace({"pipeline_state.q": y0, "obs": y0})
        cost, X = prob.rollout(x0, u)
        return cost, X.obs

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

        # Set the initial state
        rng, state_rng = jax.random.split(rng)
        x0 = prob.env.reset(state_rng)

        # Do annealed langevin sampling
        rng, langevin_rng = jax.random.split(rng)
        U, data = annealed_langevin_sample(
            options=options,
            y0=x0.obs,
            controls=U_guess,
            score_fn=lambda y, u, sigma, rng: net.apply(
                params, y, u, jnp.array([sigma])
            ),
            rng=langevin_rng,
        )

        return U, data

    # Optimize from a bunch of initial guesses
    num_samples = 32
    rng, opt_rng = jax.random.split(rng)
    opt_rng = jax.random.split(opt_rng, num_samples)
    st = time.time()
    _, data = jax.vmap(optimize_control_tape)(opt_rng)
    print(f"Sample generation took {time.time() - st:.2f} seconds")

    y0 = data.y0
    U = data.U
    sigma = data.sigma
    costs, Xs = jax.vmap(jax.vmap(rollout_from_obs))(y0, U)
    print(f"Cost: {jnp.mean(costs[-1]):.4f} +/- {jnp.std(costs[-1]):.4f}")

    # Plot the sampled trajectories
    if plot:
        prob.env.plot_scenario()
        for i in range(num_samples):
            plt.plot(
                Xs[i, -1, :, 0], Xs[i, -1, :, 1], "o-", color="blue", alpha=0.5
            )
        plt.show()

    # Animate the trajectory generation process
    if animate:
        fig, ax = plt.subplots()
        prob.env.plot_scenario()
        path = ax.plot([], [], "o-")[0]

        def update(i: int):
            j, i = divmod(i, options.num_noise_levels)
            j = j % num_samples
            ax.set_title(f"σₖ={sigma[0, i, 0, 0]:.4f}")
            path.set_data(Xs[j, i, :, 0], Xs[j, i, :, 1])
            return path

        anim = FuncAnimation(  # noqa: F841 anim must stay in scope until plot
            fig,
            update,
            frames=options.num_noise_levels * num_samples,
            interval=10,
        )
        plt.show()


if __name__ == "__main__":
    usage = "Usage: python reach_avoid.py [generate|fit|deploy|gd|animate]"

    if len(sys.argv) != 2:
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
