import pickle
import sys
import time

import jax
import mujoco
import mujoco.viewer

from rddp.architectures import ScoreMLP
from rddp.envs.pendulum import PendulumSwingupEnv
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.ocp import OptimalControlProblem
from rddp.policy import DiffusionPolicy
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions

# Global planning horizon definition
HORIZON = 50


def generate_dataset() -> None:
    """Generate a pendulum swingup dataset."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/pendulum/"

    prob = OptimalControlProblem(PendulumSwingupEnv(), num_steps=HORIZON)
    langevin_options = AnnealedLangevinOptions(
        denoising_steps=1000,
        starting_noise_level=0.1,
        step_size=0.1,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=256,
        num_rollouts_per_data_point=128,
        save_every=50,
        print_every=50,
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
    data_dir = "/tmp/pendulum/"

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
    net = ScoreMLP(layer_sizes=(512,) * 3)

    # Train the score network
    st = time.time()
    params, metrics = train(net, data_dir + "dataset.h5", training_options)
    print(f"Training took {time.time() - st:.2f} seconds")

    # Save the trained policy
    policy = DiffusionPolicy(net, params, langevin_options, (HORIZON - 1, 1))
    fname = "/tmp/pendulum/policy.pkl"
    policy.save(fname)
    print(f"Saved trained model to {fname}")


def deploy_trained_model() -> None:
    """Deploy the trained score model."""
    rng = jax.random.PRNGKey(0)
    prob = OptimalControlProblem(PendulumSwingupEnv(), num_steps=HORIZON)
    policy = DiffusionPolicy.load("/tmp/pendulum/policy.pkl")

    # Jit some helper functions
    jit_reset = jax.jit(prob.env.reset)
    jit_rollout = jax.jit(prob.rollout)
    jit_policy = jax.jit(policy.apply)

    rng, reset_rng, policy_rng = jax.random.split(rng, 3)
    x0 = jit_reset(reset_rng)
    U = jit_policy(x0.obs, policy_rng)
    cost, states = jit_rollout(x0, U)
    print(f"Total cost: {cost}")

    # Visualize the trajectory
    mj_model = prob.env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    dt = float(prob.env.dt)
    t = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            # Update the position
            mj_data.qpos = states.pipeline_state.q[t]
            mj_data.qvel = states.pipeline_state.qd[t]
            mj_data.ctrl = U[t]

            # Update the viewer
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Try to run in realtime
            elapsed_time = time.time() - start_time
            if elapsed_time < dt:
                time.sleep(dt - elapsed_time)

            t += 1
            if t >= HORIZON:
                time.sleep(1.0)
                t = 0


if __name__ == "__main__":
    usage = "Usage: python pendulum.py [generate|fit|deploy]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == "generate":
        generate_dataset()
    elif sys.argv[1] == "fit":
        fit_score_model()
    elif sys.argv[1] == "deploy":
        deploy_trained_model()
    else:
        print(usage)
        sys.exit(1)
