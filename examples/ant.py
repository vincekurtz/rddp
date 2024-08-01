import pickle
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from brax.envs.ant import Ant

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
    save_path = "/tmp/ant/"

    prob = OptimalControlProblem(Ant(), num_steps=HORIZON)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=30,
        starting_noise_level=0.1,
        num_steps=1,
        step_size=1.0,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=1,
        num_rollouts_per_data_point=8,
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
    data_dir = "/tmp/ant/"

    # Load the langiven sampling options
    with open(data_dir + "langevin_options.pkl", "rb") as f:
        langevin_options = pickle.load(f)

    # Set up the training options and the score network
    training_options = TrainingOptions(
        batch_size=30,
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
    fname = "/tmp/ant_score_model.pkl"
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
    prob = OptimalControlProblem(Ant(), num_steps=HORIZON)

    def rollout_from_obs(y0: jnp.ndarray, u: jnp.ndarray):
        """Do a rollout from an observation, and return observations."""
        q = prob.env.sys.init_q
        qd = jnp.zeros((prob.env.sys.nv,))
        x0 = prob.env.reset(rng)
        x0 = x0.tree_replace(
            {"pipeline_state.q": q, "pipeline_state.qd": qd, "obs": y0}
        )
        cost, X = prob.rollout(x0, u)
        return cost, X.obs

    # Load the trained score network
    with open("/tmp/ant_score_model.pkl", "rb") as f:
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
            guess_rng, (prob.num_steps - 1, prob.env.action_size)
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

    y0 = data.y0[:, -1, -1, :]  # x0, noise level, step, dim
    U = data.U[:, -1, -1, :, :]
    costs, Xs = jax.vmap(rollout_from_obs)(y0, U)
    print(f"Cost: {jnp.mean(costs):.4f} +/- {jnp.std(costs):.4f}")

    nq = prob.env.sys.nq
    visualize_trajectories(prob, Xs[:, :, :nq])


def visualize_trajectories(prob: OptimalControlProblem, q: jnp.ndarray) -> None:
    """Visualize optimized trajectories on the mujoco viewer.

    Args:
        prob: The optimal control problem, with an MJX env.
        q: The optimized positions, size (num_samples, HORIZON, nq).
    """
    num_samples = q.shape[0]

    mj_model = prob.env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    dt = float(prob.env.dt)
    i, t = 0, 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            # Update the position of the cart-pole
            mj_data.qpos[:] = q[i, t]
            mj_data.qvel[:] = 0.0

            # Update the viewer
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Try to run in realtime
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

            # Advance the time step and/or the sample index
            t += 1
            if t == HORIZON:
                time.sleep(1.0)
                t = 0
                i += 1
                i = i % num_samples


if __name__ == "__main__":
    usage = "Usage: python ant.py [generate|fit|deploy]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == "generate":
        generate_dataset()
    elif sys.argv[1] == "fit":
        fit_score_model()
    elif sys.argv[1] == "deploy":
        deploy_trained_model()
