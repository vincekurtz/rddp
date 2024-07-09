import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.tasks.reach_avoid import ReachAvoid
from rddp.utils import AnnealedLangevinOptions, DiffusionDataset


def test_score_estimate() -> None:
    """Test our numerical score estimation."""
    rng = jax.random.PRNGKey(0)

    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=3,
        starting_noise_level=0.1,
        num_steps=4,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        noise_levels_per_file=1,
        temperature=0.1,
        num_initial_states=1,
        num_rollouts_per_data_point=10,
    )
    generator = DatasetGenerator(
        prob, langevin_options, gen_options, "/dev/null"
    )

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
        noise_levels_per_file=50,
        temperature=0.01,
        num_initial_states=5,
        num_rollouts_per_data_point=16,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options, local_dir)

    # Generate and save the dataset
    rng = jax.random.PRNGKey(0)
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng)

    # Load one of the saved files
    with open(local_dir / "diffusion_data_1.pkl", "rb") as f:
        loaded = pickle.load(f)

    assert isinstance(loaded, DiffusionDataset)
    # Check sizes
    Nx = gen_options.num_initial_states
    K = gen_options.noise_levels_per_file
    N = langevin_options.num_steps

    assert loaded.x0.shape == (Nx * K * N, 2)
    assert loaded.U.shape == (Nx * K * N, prob.num_steps - 1, 2)
    assert loaded.s.shape == (Nx * K * N, prob.num_steps - 1, 2)
    assert loaded.k.shape == (Nx * K * N, 1)
    assert loaded.sigma.shape == (Nx * K * N, 1)

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_score_estimate()
    test_generate()
