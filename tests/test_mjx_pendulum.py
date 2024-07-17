from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
from mujoco import mjx

from rddp.architectures import ScoreMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.systems.mjx_pendulum import MjxPendulum
from rddp.tasks.mjx_pendulum_swingup import MjxPendulumSwingup
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions, HDF5DiffusionDataset


def test_pendulum() -> None:
    """Test the basic features of the MJX pendulum dynamics."""
    sys = MjxPendulum()

    assert sys.state_shape == (2,)
    assert sys.action_shape == (1,)
    assert sys.observation_shape == (3,)

    data = sys.make_data()
    assert isinstance(data, mjx.Data)

    data = data.replace(qpos=jnp.array([jnp.pi / 2]), qvel=jnp.array([0.0]))
    u = jnp.array([0.0])

    old_theta, old_theta_dot = data.qpos[0], data.qvel[0]
    data = sys.f(data, u)
    assert data.qpos.shape == (1,)
    assert data.qvel.shape == (1,)
    new_theta, new_theta_dot = data.qpos[0], data.qvel[0]

    assert new_theta != old_theta
    assert new_theta_dot != old_theta_dot

    y = sys.g(data)
    y_pred = jnp.array([jnp.cos(new_theta), jnp.sin(new_theta), new_theta_dot])
    assert y.shape == (3,)
    assert jnp.allclose(y, y_pred)


def test_rollout() -> None:
    """Test that we can roll out the pendulum dynamics."""
    sys = MjxPendulum()
    data = sys.make_data()
    data = data.replace(qpos=jnp.array([jnp.pi / 2]), qvel=jnp.array([0.0]))

    control_tape = jnp.zeros((10, 1))
    all_data = sys.rollout(control_tape, data)
    assert all_data.qpos.shape == (10, 1)
    assert all_data.qvel.shape == (10, 1)


def test_train() -> None:
    """Test training a simple policy with an MJX-based system."""
    rng = jax.random.PRNGKey(0)

    # Make a temporary directory for the dataset
    local_dir = Path("_test_mjx_training")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Generate a training dataset
    prob = MjxPendulumSwingup(num_steps=5)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=100,
        starting_noise_level=0.5,
        num_steps=10,
        step_size=0.01,
    )
    gen_options = DatasetGenerationOptions(
        temperature=0.001,
        num_initial_states=256,
        num_rollouts_per_data_point=8,
        save_every=100,
        save_path=local_dir,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng)
    assert (local_dir / "dataset.h5").exists()

    # Train a score network
    options = TrainingOptions(
        batch_size=512,
        num_superbatches=1,
        epochs=4,
        learning_rate=1e-3,
    )
    net = ScoreMLP(layer_sizes=(32,) * 3)

    params, metrics = train(net, local_dir / "dataset.h5", options)
    assert metrics["loss"][-1] < metrics["loss"][0]

    test_idx = 129
    with h5py.File(local_dir / "dataset.h5", "r") as f:
        h5_dataset = HDF5DiffusionDataset(f)
        dataset = h5_dataset[...]
    score_estimate = net.apply(
        params,
        dataset.y0[test_idx],
        dataset.U[test_idx],
        dataset.sigma[test_idx],
    )
    assert score_estimate.shape == dataset.s[test_idx].shape

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_pendulum()
    test_rollout()
    test_train()
