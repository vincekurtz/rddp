import pickle
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp

from rddp.envs.reach_avoid import ReachAvoidEnv
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.ocp import OptimalControlProblem
from rddp.utils import (
    AnnealedLangevinOptions,
    DiffusionDataset,
    HDF5DiffusionDataset,
)


def test_score_estimate() -> None:
    """Test our numerical score estimation."""
    rng = jax.random.PRNGKey(0)

    # Create a temporary directory
    local_dir = Path("_test_score_estimate")
    local_dir.mkdir(parents=True, exist_ok=True)

    prob = OptimalControlProblem(ReachAvoidEnv(num_steps=20), num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=3,
        starting_noise_level=0.1,
        num_steps=4,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=1,
        num_rollouts_per_data_point=10,
        save_every=1,
        save_path=local_dir,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Set initial state
    rng, reset_rng = jax.random.split(rng)
    x0 = prob.env.reset(reset_rng)

    # Guess a control sequence
    sigma = langevin_options.starting_noise_level
    rng, U_rng = jax.random.split(rng)
    U = sigma * jax.random.normal(
        U_rng, (prob.num_steps - 1, prob.env.action_size)
    )

    # Estimate the score
    rng, score_estimate_rng = jax.random.split(rng)
    s = generator.estimate_noised_score(x0, U, sigma, score_estimate_rng)

    assert s.shape == U.shape

    # Gradient descent should improve the cost
    original_cost, _ = prob.rollout(x0, U)
    new_cost, _ = prob.rollout(x0, U + sigma**2 * s)
    assert new_cost < original_cost

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_save_dataset() -> None:
    """Test saving the dataset to disk."""
    rng = jax.random.PRNGKey(0)

    # Create a temporary directory
    local_dir = Path("_test_score_estimate")
    local_dir.mkdir(parents=True, exist_ok=True)

    prob = OptimalControlProblem(ReachAvoidEnv(num_steps=20), num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=3,
        starting_noise_level=0.1,
        num_steps=4,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=1,
        num_rollouts_per_data_point=10,
        save_every=1,
        save_path=local_dir,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Check that the langevin options were saved
    with open(local_dir / "langevin_options.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert loaded == langevin_options

    # Check than an hdf5 file was created
    assert (local_dir / "dataset.h5").exists()

    # Make some random fake data and save it
    num_samples = 100
    rng, y0_rng, U_rng, s_rng, sigma_rng, k_rng = jax.random.split(rng, 6)
    y0 = jax.random.uniform(y0_rng, (num_samples, 2))
    U = jax.random.uniform(U_rng, (num_samples, 19, 2))
    s = jax.random.uniform(s_rng, (num_samples, 19, 2))
    sigma = jax.random.uniform(sigma_rng, (num_samples, 1))
    k = jax.random.randint(k_rng, (num_samples, 1), 0, 100)
    dataset = DiffusionDataset(y0=y0, U=U, s=s, sigma=sigma, k=k)
    generator.save_dataset(dataset)

    # Check that the hdf5 file was updated
    with h5py.File(local_dir / "dataset.h5", "r") as f:
        h5_dataset = HDF5DiffusionDataset(f)
        assert len(h5_dataset) == num_samples
        assert h5_dataset.y0.shape == (num_samples, 2)
        assert h5_dataset.U.shape == (num_samples, 19, 2)
        assert h5_dataset.s.shape == (num_samples, 19, 2)
        assert h5_dataset.sigma.shape == (num_samples, 1)
        assert h5_dataset.k.shape == (num_samples, 1)

        # Check that slicing on the hdf5 dataset works
        partial_dataset = h5_dataset[2:14]
        assert jnp.all(partial_dataset.y0 == y0[2:14])
        assert jnp.all(partial_dataset.U == U[2:14])
        assert jnp.all(partial_dataset.s == s[2:14])
        assert jnp.all(partial_dataset.sigma == sigma[2:14])
        assert jnp.all(partial_dataset.k == k[2:14])

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_generate() -> None:
    """Test the dataset generation process."""
    # Create a temporary directory
    local_dir = Path("_test_generate")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create a generator
    prob = OptimalControlProblem(ReachAvoidEnv(num_steps=20), num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=250,
        starting_noise_level=0.1,
        num_steps=8,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=5,
        num_rollouts_per_data_point=16,
        save_every=50,
        save_path=local_dir,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate and save the dataset
    rng = jax.random.PRNGKey(0)
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng)

    # Check that we've generated the correct number of data points
    N = (
        langevin_options.num_steps
        * langevin_options.num_noise_levels
        * gen_options.num_initial_states
    )
    with h5py.File(local_dir / "dataset.h5", "r") as f:
        h5_dataset = HDF5DiffusionDataset(f)
        assert len(h5_dataset) == N

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_score_estimate()
    test_save_dataset()
    test_generate()
