import pickle
import sys
import time

import jax
import jax.numpy as jnp

from rddp.architectures import ScoreMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.tasks.mjx_pendulum_swingup import MjxPendulumSwingup
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions, annealed_langevin_sample

# Global planning horizon definition
HORIZON = 10


def generate_dataset(plot: bool = True) -> None:
    """Generate some training data."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/pendulum"

    # Problem setup
    prob = MjxPendulumSwingup(num_steps=HORIZON)
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


def deploy_trained_model() -> None:
    """Use the trained model to generate optimal actions."""
    rng = jax.random.PRNGKey(0)
    prob = MjxPendulumSwingup(num_steps=HORIZON)

    # Load the trained score network
    with open("/tmp/pendulum_score_model.pkl", "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    net = data["net"]
    options = data["langevin_options"]

    # Decide how much noise to add in the Langevin sampling
    options = options.replace(noise_injection_level=0.0)

    def policy_fn(y0: jnp.ndarray, rng: jax.random.PRNGKey):
        """Optimize the control sequence using Langevin dynamics."""
        # Guess an initial control sequence
        rng, guess_rng = jax.random.split(rng, 2)
        U_guess = options.starting_noise_level * jax.random.normal(
            guess_rng, (prob.num_steps - 1, 1)
        )

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

        return U, data

    # Sample an initial state
    rng, init_rng = jax.random.split(rng)
    x0 = prob.sample_initial_state(init_rng)
    policy = lambda y: policy_fn(y, rng)[0][0]  # fixed RNG
    prob.sys.simulate_and_render(
        x0, policy, num_steps=jnp.inf, fixedcamid=0, cam_type=2
    )


if __name__ == "__main__":
    usage = "Usage: python mjx_pendulum.py [generate|fit|deploy|gd|animate]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)
    elif sys.argv[1] == "generate":
        generate_dataset()
    elif sys.argv[1] == "fit":
        fit_score_model()
    elif sys.argv[1] == "deploy":
        deploy_trained_model()
    else:
        print(usage)
        sys.exit(1)
