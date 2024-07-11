import pickle
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp

from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.tasks.reach_avoid import ReachAvoid
from rddp.utils import AnnealedLangevinOptions, DiffusionDataset


def test_score_estimate() -> None:
    """Test our numerical score estimation."""
    rng = jax.random.PRNGKey(0)

    # Create a temporary directory
    local_dir = Path("_test_score_estimate")
    local_dir.mkdir(parents=True, exist_ok=True)

    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=3,
        starting_noise_level=0.1,
        num_steps=4,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        temperature=0.1,
        num_initial_states=1,
        num_rollouts_per_data_point=10,
        save_every=1,
        save_path=local_dir,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Set initial state
    x0 = jnp.array([0.1, -1.5])

    # Guess a control sequence
    sigma = langevin_options.starting_noise_level
    rng, U_rng = jax.random.split(rng)
    U = sigma * jax.random.normal(
        U_rng, (prob.num_steps - 1, prob.sys.action_shape[0])
    )

    # Estimate the score
    rng, score_estimate_rng = jax.random.split(rng)
    s = generator.estimate_noised_score(x0, U, sigma, score_estimate_rng)

    assert s.shape == U.shape

    # Gradient descent should improve the cost
    assert prob.total_cost(U, x0) > prob.total_cost(U + sigma**2 * s, x0)

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

    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=3,
        starting_noise_level=0.1,
        num_steps=4,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        temperature=0.1,
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
    rng, x0_rng, U_rng, s_rng, sigma_rng, k_rng = jax.random.split(rng, 6)
    x0 = jax.random.uniform(x0_rng, (num_samples, 2))
    U = jax.random.uniform(U_rng, (num_samples, 19, 2))
    s = jax.random.uniform(s_rng, (num_samples, 19, 2))
    sigma = jax.random.uniform(sigma_rng, (num_samples, 1))
    k = jax.random.randint(k_rng, (num_samples, 1), 0, 100)
    dataset = DiffusionDataset(x0=x0, U=U, s=s, sigma=sigma, k=k)
    generator.save_dataset(dataset)

    # Check that the hdf5 file was updated
    with h5py.File(local_dir / "dataset.h5", "r") as f:
        x0, U, s, sigma, k = f["x0"], f["U"], f["s"], f["sigma"], f["k"]
        assert x0.shape == (num_samples, 2)
        assert U.shape == (num_samples, 19, 2)
        assert s.shape == (num_samples, 19, 2)
        assert sigma.shape == (num_samples, 1)
        assert k.shape == (num_samples, 1)

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_generate() -> None:
    """Test the dataset generation process."""
    # Create a temporary directory
    local_dir = Path("_test_save_dataset")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create a generator
    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=250,
        starting_noise_level=0.1,
        num_steps=8,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        temperature=0.01,
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
        x0, U, s, sigma, k = f["x0"], f["U"], f["s"], f["sigma"], f["k"]
        assert x0.shape[0] == N
        assert U.shape[0] == N
        assert s.shape[0] == N
        assert sigma.shape[0] == N
        assert k.shape[0] == N

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_score_estimate()
    test_save_dataset()
    test_generate()
