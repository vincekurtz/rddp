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
    annealed_langevin_sample,
)


def test_score_estimate() -> None:
    """Test our numerical score estimation."""
    rng = jax.random.PRNGKey(0)

    # Create a temporary directory
    local_dir = Path("_test_score_estimate")
    local_dir.mkdir(parents=True, exist_ok=True)

    prob = OptimalControlProblem(ReachAvoidEnv(num_steps=20), num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=10,
        starting_noise_level=0.1,
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
    s, cost, obs = generator.estimate_noised_score(
        x0, U, sigma, score_estimate_rng
    )

    assert s.shape == U.shape
    assert cost.shape == ()
    assert obs.shape == (prob.num_steps, prob.env.observation_size)

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
        num_noise_levels=10,
        starting_noise_level=0.1,
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
    rng, Y_rng, U_rng, s_rng, sigma_rng, k_rng, c_rng = jax.random.split(rng, 7)
    Y = jax.random.uniform(Y_rng, (num_samples, 20, 2))
    U = jax.random.uniform(U_rng, (num_samples, 19, 2))
    s = jax.random.uniform(s_rng, (num_samples, 19, 2))
    sigma = jax.random.uniform(sigma_rng, (num_samples, 1))
    k = jax.random.randint(k_rng, (num_samples, 1), 0, 100)
    cost = jax.random.uniform(c_rng, (num_samples, 1))
    dataset = DiffusionDataset(Y=Y, U=U, s=s, sigma=sigma, k=k, cost=cost)
    generator.save_dataset(dataset)

    # Check that the hdf5 file was updated
    with h5py.File(local_dir / "dataset.h5", "r") as f:
        h5_dataset = HDF5DiffusionDataset(f)
        assert len(h5_dataset) == num_samples
        assert h5_dataset.Y.shape == (num_samples, 20, 2)
        assert h5_dataset.U.shape == (num_samples, 19, 2)
        assert h5_dataset.s.shape == (num_samples, 19, 2)
        assert h5_dataset.sigma.shape == (num_samples, 1)
        assert h5_dataset.k.shape == (num_samples, 1)
        assert h5_dataset.cost.shape == (num_samples, 1)

        # Check that slicing on the hdf5 dataset works
        partial_dataset = h5_dataset[2:14]
        assert jnp.all(partial_dataset.Y == Y[2:14])
        assert jnp.all(partial_dataset.U == U[2:14])
        assert jnp.all(partial_dataset.s == s[2:14])
        assert jnp.all(partial_dataset.sigma == sigma[2:14])
        assert jnp.all(partial_dataset.k == k[2:14])
        assert jnp.all(partial_dataset.cost == cost[2:14])

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
        num_noise_levels=2000,
        starting_noise_level=0.1,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=5,
        num_rollouts_per_data_point=16,
        save_path=local_dir,
        save_every=500,
        print_every=100,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate and save the dataset
    rng = jax.random.PRNGKey(0)
    rng, gen_rng = jax.random.split(rng)
    generator.generate(gen_rng)

    # Check that we've generated the correct number of data points
    N = langevin_options.num_noise_levels * gen_options.num_initial_states
    with h5py.File(local_dir / "dataset.h5", "r") as f:
        h5_dataset = HDF5DiffusionDataset(f)
        assert len(h5_dataset) == N
        dataset = h5_dataset[:]

    # Check that the costs and outputs that we generated match what we get
    # from manual rollouts
    y0 = dataset.Y[:, 0]
    rng, reset_rng = jax.random.split(rng)
    reset_rng = jax.random.split(reset_rng, N)
    x0 = jax.jit(jax.vmap(prob.env.reset))(reset_rng)
    x0 = x0.tree_replace({"pipeline_state.q": y0, "obs": y0})
    assert jnp.all(x0.obs == y0)
    assert jnp.all(x0.pipeline_state.q == y0)

    costs, states = jax.jit(jax.vmap(prob.rollout))(x0, dataset.U)
    assert jnp.allclose(costs[None].T, dataset.cost)
    assert jnp.allclose(states.obs, dataset.Y)

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_langevin() -> None:
    """Check that we get the same results as utils.annealed_langevin_sample."""
    # Create a temporary directory
    local_dir = Path("_test_langevin")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Set up a problem instance
    prob = OptimalControlProblem(ReachAvoidEnv(num_steps=10), num_steps=10)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=3000,
        starting_noise_level=0.1,
        step_size=0.05,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=1,
        num_rollouts_per_data_point=128,
        save_path=local_dir,
        save_every=1000,
        print_every=500,
    )

    # Do langevin sampling with our generator
    rng = jax.random.PRNGKey(0)
    rng, gen_rng = jax.random.split(rng)
    generator = DatasetGenerator(prob, langevin_options, gen_options)
    generator.generate(gen_rng)

    with h5py.File(local_dir / "dataset.h5", "r") as f:
        h5_dataset = HDF5DiffusionDataset(f)
        U_gen = jnp.array(h5_dataset.U)
        y0 = h5_dataset.Y[-1, 0]
        generated_dataset = h5_dataset[:]

    # Do langevin sampling with the utils function
    # N.B. the awkward series of rng splits ensures that we get the same
    # random seed for the two methods.
    rng = jax.random.PRNGKey(0)
    rng, langevin_rng = jax.random.split(rng)
    rng, init_rng = jax.random.split(langevin_rng)
    init_rng = jax.random.split(init_rng, 1)[0]
    x0 = prob.env.reset(init_rng)
    assert jnp.allclose(x0.obs, y0), "Initial states do not match, check rng"

    rng, init_rng = jax.random.split(rng)
    U = langevin_options.starting_noise_level * jax.random.normal(
        init_rng, (prob.num_steps - 1, prob.env.action_size)
    )
    assert jnp.allclose(U, U_gen[0]), "Initial controls do not match, check rng"

    def score_fn(x, u, sigma, rng):  # noqa: ANN001 (skip type annotations)
        return generator.estimate_noised_score(x, u, sigma, rng)[0]

    U, langevin_dataset = annealed_langevin_sample(
        langevin_options, x0, U, score_fn, rng
    )

    langevin_cost = prob.rollout(x0, U)[0]
    generated_cost = prob.rollout(x0, U_gen[-1])[0]

    assert jnp.allclose(langevin_cost, generated_cost, atol=1e-2)

    assert langevin_dataset.s.shape == generated_dataset.s.shape
    assert langevin_dataset.U.shape == generated_dataset.U.shape
    assert langevin_dataset.sigma.shape == generated_dataset.sigma.shape
    assert langevin_dataset.k.shape == generated_dataset.k.shape

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_score_estimate()
    test_save_dataset()
    test_generate()
    test_langevin()
