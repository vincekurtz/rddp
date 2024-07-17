import pickle
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.architectures import ScoreMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.gradient_descent import solve as solve_gd
from rddp.tasks.pendulum_swingup import PendulumSwingup
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions, annealed_langevin_sample

# Global planning horizon definition
HORIZON = 10


def solve_with_gradient_descent() -> None:
    """Solve the optimal control problem using simple gradient descent."""
    prob = PendulumSwingup(num_steps=HORIZON)
    x0 = jnp.array([3.0, 2.0])
    U, _, _ = solve_gd(prob, x0)

    prob.plot_scenario()
    X = prob.sys.rollout(U, x0)
    plt.plot(X[:, 0], X[:, 1], "o-")
    plt.show()


def generate_dataset(plot: bool = True) -> None:
    """Generate some training data."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/pendulum"

    # Problem setup
    prob = PendulumSwingup(num_steps=HORIZON)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=300,
        starting_noise_level=1.0,
        num_steps=100,
        step_size=0.01,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        temperature=0.01,
        num_initial_states=256,
        num_rollouts_per_data_point=128,
        save_every=100,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    st = time.time()
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng)
    print(f"Data generation took {time.time() - st:.2f} seconds")


def fit_score_model() -> None:
    """Fit a simple score model to the generated data."""
    # Specify location of the training data
    data_dir = "/tmp/pendulum/"

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
    fname = "/tmp/pendulum_score_model.pkl"
    with open(fname, "wb") as f:
        data = {
            "params": params,
            "net": net,
            "langevin_options": langevin_options,
        }
        pickle.dump(data, f)
    print(f"Saved trained model to {fname}")


def deploy_trained_model(plot: bool = True) -> None:
    """Use the trained model to generate optimal actions."""
    rng = jax.random.PRNGKey(0)
    prob = PendulumSwingup(num_steps=HORIZON)

    # Load the trained score network
    with open("/tmp/pendulum_score_model.pkl", "rb") as f:
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
            guess_rng, (prob.num_steps - 1, 1)
        )

        # Set the initial observation
        rng, state_rng = jax.random.split(rng)
        x0 = prob.sample_initial_state(state_rng)
        y0 = prob.sys.g(x0)

        # Do annealed langevin sampling
        rng, langevin_rng = jax.random.split(rng)
        U, data = annealed_langevin_sample(
            options=options,
            y0=y0,
            u_init=U_guess,
            score_fn=lambda x, u, sigma, rng: net.apply(
                params, x, u, jnp.array([sigma])
            ),
            rng=langevin_rng,
        )

        return U, x0, data

    # Optimize from a bunch of initial guesses
    num_samples = 32
    rng, opt_rng = jax.random.split(rng)
    opt_rng = jax.random.split(opt_rng, num_samples)
    st = time.time()
    Us, x0s, data = jax.vmap(optimize_control_tape)(opt_rng)
    print(f"Sample generation took {time.time() - st:.2f} seconds")
    Xs = jax.vmap(prob.sys.rollout)(Us, x0s)
    costs = jax.vmap(prob.total_cost)(Us, x0s)
    print(f"Cost: {jnp.mean(costs):.4f} +/- {jnp.std(costs):.4f}")

    # Plot the sampled trajectories
    if plot:
        prob.plot_scenario()
        for i in range(num_samples):
            plt.plot(Xs[i, :, 0], Xs[i, :, 1], "o-", color="blue", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    usage = "Usage: python pendulum.py [generate|fit|deploy|gd|animate]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)
    if sys.argv[1] == "gd":
        solve_with_gradient_descent()
    elif sys.argv[1] == "generate":
        generate_dataset()
    elif sys.argv[1] == "fit":
        fit_score_model()
    elif sys.argv[1] == "deploy":
        deploy_trained_model()
    else:
        print(usage)
        sys.exit(1)
