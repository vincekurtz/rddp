import pickle
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from brax.envs.inverted_pendulum import InvertedPendulum

from rddp.architectures import ScoreMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.ocp import OptimalControlProblem
from rddp.policy import DiffusionPolicy
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions

# Global planning horizon definition
HORIZON = 30


def generate_dataset() -> None:
    """Generate a cart-pole swingup dataset."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/cart_pole/"

    prob = OptimalControlProblem(InvertedPendulum(), num_steps=HORIZON)
    langevin_options = AnnealedLangevinOptions(
        denoising_steps=100,
        starting_noise_level=0.1,
        step_size=0.1,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=256,
        num_rollouts_per_data_point=128,
        save_every=20,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate some data
    st = time.time()
    rng, gen_rng = jax.random.split(rng)
    generator.generate(gen_rng)
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
        batch_size=1280,
        num_superbatches=1,
        epochs=500,
        print_every=100,
        learning_rate=1e-3,
    )
    net = ScoreMLP(layer_sizes=(128,) * 3)

    # Train the score network
    st = time.time()
    params, metrics = train(net, data_dir + "dataset.h5", training_options)
    print(f"Training took {time.time() - st:.2f} seconds")

    # Save the trained policy
    policy = DiffusionPolicy(net, params, langevin_options, (HORIZON - 1, 1))
    fname = "/tmp/cart_pole_policy.pkl"
    policy.save(fname)
    print(f"Saved trained model to {fname}")


def deploy_trained_model() -> None:
    """Deploy the trained score model."""
    rng = jax.random.PRNGKey(0)
    prob = OptimalControlProblem(InvertedPendulum(), num_steps=HORIZON)
    policy = DiffusionPolicy.load("/tmp/cart_pole_policy.pkl")

    # Jit some helper functions
    jit_reset = jax.jit(prob.env.reset)
    jit_rollout = jax.jit(prob.rollout)
    jit_policy = jax.jit(policy.apply)

    rng, reset_rng, policy_rng = jax.random.split(rng, 3)
    x0 = jit_reset(reset_rng)
    U = jit_policy(x0.obs, policy_rng)
    cost, states = jit_rollout(x0, U)
    print(f"Total cost: {cost}")

    visualize_trajectory(prob, states.pipeline_state.q)


def visualize_trajectory(prob: OptimalControlProblem, q: jnp.ndarray) -> None:
    """Visualize optimized trajectories on the mujoco viewer.

    Args:
        prob: The optimal control problem, with an MJX env.
        q: The optimized positions, size (num_steps, nq).
    """
    num_steps = q.shape[0]

    mj_model = prob.env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    dt = float(prob.env.dt)
    t = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            # Update the position of the cart-pole
            mj_data.qpos[:2] = q[t]
            mj_data.qvel[:2] = 0.0

            # Update the viewer
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Try to run in realtime
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

            # Advance the time step
            t += 1
            if t == num_steps:
                time.sleep(1.0)
                t = 0


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
