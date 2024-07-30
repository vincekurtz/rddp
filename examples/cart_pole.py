import pickle
import sys
import time

import jax
import jax.numpy as jnp
from brax.envs.inverted_pendulum import InvertedPendulum

from rddp.architectures import ScoreMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.ocp import OptimalControlProblem
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions, annealed_langevin_sample

# Global planning horizon definition
HORIZON = 20


def generate_dataset() -> None:
    """Generate a cart-pole swingup dataset."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/cart_pole/"

    prob = OptimalControlProblem(InvertedPendulum(), num_steps=HORIZON)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=30,
        starting_noise_level=0.1,
        num_steps=1,
        step_size=1.0,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=128,
        num_rollouts_per_data_point=128,
        save_every=1,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate some data
    st = time.time()
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng)
    print(f"Data generation took {time.time() - st:.2f} seconds")


def fit_score_model() -> None:
    """Fit a simple score model to the generated data."""
    # Specify location of the training data
    data_dir = "/tmp/cart_pole/"

    # Load the langiven sampling options
    with open(data_dir + "langevin_options.pkl", "rb") as f:
        langevin_options = pickle.load(f)

    # Set up the training options and the score network
    training_options = TrainingOptions(
        batch_size=3840,
        num_superbatches=1,
        epochs=500,
        learning_rate=1e-3,
    )
    net = ScoreMLP(layer_sizes=(128,) * 3)

    # Train the score network
    st = time.time()
    params, metrics = train(net, data_dir + "dataset.h5", training_options)
    print(f"Training took {time.time() - st:.2f} seconds")

    # Save the trained model and parameters
    fname = "/tmp/cart_pole_score_model.pkl"
    with open(fname, "wb") as f:
        data = {
            "params": params,
            "net": net,
            "langevin_options": langevin_options,
        }
        pickle.dump(data, f)
    print(f"Saved trained model to {fname}")


def deploy_trained_model() -> None:
    """Deploy the trained score model."""
    rng = jax.random.PRNGKey(0)
    prob = OptimalControlProblem(InvertedPendulum(), num_steps=HORIZON)

    def rollout_from_obs(y0: jnp.ndarray, u: jnp.ndarray):
        """Do a rollout from an observation, and return observations."""
        x0 = prob.env.reset(rng)
        x0 = x0.tree_replace(
            {"pipeline_state.q": y0[:2], "pipeline_state.qd": y0[2:], "obs": y0}
        )
        cost, X = prob.rollout(x0, u)
        return cost, X.obs

    # Load the trained score network
    with open("/tmp/cart_pole_score_model.pkl", "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    net = data["net"]
    options = data["langevin_options"]
    print("Loaded trained model")

    # Decide how much noise to add in the Langevin sampling
    options = options.replace(noise_injection_level=0.0)

    def optimize_control_tape(rng: jax.random.PRNGKey):
        """Optimize the control sequence using Langevin dynamics."""
        # Guess an initial control sequence
        rng, guess_rng = jax.random.split(rng, 2)
        U_guess = options.starting_noise_level * jax.random.normal(
            guess_rng, (prob.num_steps - 1, 1)
        )

        # Set the initial state
        rng, state_rng = jax.random.split(rng)
        x0 = prob.env.reset(state_rng)

        # Do annealed langevin sampling
        rng, langevin_rng = jax.random.split(rng)
        U, data = annealed_langevin_sample(
            options=options,
            y0=x0.obs,
            u_init=U_guess,
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

    y0 = data.y0[:, :, -1, :]  # take the last sample at each noise level
    U = data.U[:, :, -1, :]
    costs, Xs = jax.vmap(jax.vmap(rollout_from_obs))(y0, U)
    print(f"Cost: {jnp.mean(costs[-1]):.4f} +/- {jnp.std(costs[-1]):.4f}")


if __name__ == "__main__":
    usage = "Usage: python cart_pole.py [generate|fit|deploy]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == "generate":
        generate_dataset()
    elif sys.argv[1] == "fit":
        fit_score_model()
    elif sys.argv[1] == "deploy":
        deploy_trained_model()
