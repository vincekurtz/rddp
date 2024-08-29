import pickle
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.architectures import ScoreMLP
from rddp.envs.double_integrator import DoubleIntegratorEnv
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.gradient_descent import solve as solve_gd
from rddp.ocp import OptimalControlProblem
from rddp.policy import DiffusionPolicy
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions

# Global planning horizon definition
HORIZON = 10


def solve_with_gradient_descent() -> None:
    """Solve the optimal control problem using simple gradient descent."""
    rng = jax.random.PRNGKey(1)
    prob = OptimalControlProblem(DoubleIntegratorEnv(), HORIZON, u_max=10.0)

    rng, reset_rng = jax.random.split(rng)
    x0 = prob.env.reset(reset_rng)
    U, _, _ = solve_gd(prob, x0, max_iter=5000, print_every=500)

    prob.env.plot_scenario()
    _, states = prob.rollout(x0, U)
    X = jnp.array([states.obs[i] for i in range(HORIZON + 1)])
    plt.plot(X[:, 0], X[:, 1], "o-")
    plt.show()


def generate_dataset(plot: bool = True) -> None:
    """Generate some training data."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/double_integrator"

    # Problem setup
    prob = OptimalControlProblem(
        DoubleIntegratorEnv(), num_steps=HORIZON, u_max=10.0
    )
    langevin_options = AnnealedLangevinOptions(
        denoising_steps=500,
        starting_noise_level=0.5,
        step_size=0.1,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=256,
        num_rollouts_per_data_point=128,
        save_every=100,
        print_every=100,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    st = time.time()
    rng, gen_rng = jax.random.split(rng)
    generator.generate(gen_rng)
    print(f"Data generation took {time.time() - st:.2f} seconds")


def fit_score_model() -> None:
    """Fit a simple score model to the generated data."""
    # Specify location of the training data
    data_dir = "/tmp/double_integrator/"

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

    # Save the trained policy
    fname = "/tmp/double_integrator_policy.pkl"
    policy = DiffusionPolicy(net, params, langevin_options, (HORIZON - 1, 1))
    policy.save(fname)


def deploy_trained_model(plot: bool = True) -> None:
    """Use the trained model to generate optimal actions."""
    rng = jax.random.PRNGKey(0)
    prob = OptimalControlProblem(
        DoubleIntegratorEnv(), num_steps=HORIZON, u_max=10.0
    )
    policy = DiffusionPolicy.load("/tmp/double_integrator_policy.pkl")

    def _rollout_policy(rng: jax.random.PRNGKey) -> jnp.ndarray:
        x0 = prob.env.reset(rng)
        U = policy.apply(x0.obs, rng)
        return prob.rollout(x0, U)

    num_samples = 32
    rng, sample_rng = jax.random.split(rng)
    sample_rng = jax.random.split(sample_rng, num_samples)

    costs, states = jax.vmap(_rollout_policy)(sample_rng)
    print(f"Cost: {jnp.mean(costs[-1]):.4f} +/- {jnp.std(costs[-1]):.4f}")

    # Plot the sampled trajectories
    if plot:
        prob.env.plot_scenario()
        y = states.obs
        for i in range(num_samples):
            plt.plot(y[i, :, 0], y[i, :, 1], "o-", color="blue", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    usage = "Usage: python double_integrator.py [generate|fit|deploy|gd]"

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
        deploy_trained_model(plot=True)
    else:
        print(usage)
        sys.exit(1)
