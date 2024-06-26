import jax
import jax.numpy as jnp

from rddp.data_generation import (
    AnnealedLangevinOptions,
    DatasetGenerationOptions,
    DatasetGenerator,
    DiffusionDataset,
)
from rddp.tasks.reach_avoid import ReachAvoid


def test_score_estimate() -> None:
    """Test our numerical score estimation."""
    rng = jax.random.PRNGKey(0)

    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        temperature=0.1,
        num_noise_levels=3,
        starting_noise_level=0.1,
        noise_decay_rate=0.9,
    )
    gen_options = DatasetGenerationOptions(
        num_initial_states=1,
        num_data_points_per_initial_state=1,
        num_rollouts_per_data_point=10,
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
    assert prob.total_cost(U, x0) > prob.total_cost(U + s, x0)


def test_gen_from_state() -> None:
    """Test dataset generation from a single initial state."""
    rng = jax.random.PRNGKey(0)

    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        temperature=0.01,
        num_noise_levels=250,
        starting_noise_level=0.1,
        noise_decay_rate=0.97,
    )
    gen_options = DatasetGenerationOptions(
        num_initial_states=1,
        num_data_points_per_initial_state=64,
        num_rollouts_per_data_point=16,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Set the initial state
    x0 = jnp.array([0.1, -1.5])

    # Generate the dataset
    rng, gen_rng = jax.random.split(rng)
    dataset = generator.generate_from_state(x0, gen_rng)
    assert isinstance(dataset, DiffusionDataset)

    # Check sizes
    L = langevin_options.num_noise_levels
    N = gen_options.num_data_points_per_initial_state
    assert dataset.x0.shape == (L, N, 2)
    assert dataset.U.shape == (L, N, prob.num_steps - 1, 2)
    assert dataset.s.shape == (L, N, prob.num_steps - 1, 2)
    assert dataset.k.shape == (L, N)

    # Check that the costs decreased
    costs = jax.vmap(jax.vmap(prob.total_cost))(dataset.U, dataset.x0)
    start_costs = costs[0, ...]
    final_costs = costs[-1, ...]
    assert jnp.all(final_costs < start_costs)


def test_generate() -> None:
    """Test the main dataset generation function."""
    rng = jax.random.PRNGKey(0)

    prob = ReachAvoid(num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        temperature=0.01,
        num_noise_levels=250,
        starting_noise_level=0.1,
        noise_decay_rate=0.97,
    )
    gen_options = DatasetGenerationOptions(
        num_initial_states=3,
        num_data_points_per_initial_state=8,
        num_rollouts_per_data_point=12,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    rng, gen_rng = jax.random.split(rng)
    dataset = generator.generate(gen_rng)
    assert isinstance(dataset, DiffusionDataset)

    # Check sizes
    Nx = gen_options.num_initial_states
    L = langevin_options.num_noise_levels
    N = gen_options.num_data_points_per_initial_state

    assert dataset.x0.shape == (Nx * L * N, 2)
    assert dataset.U.shape == (Nx * L * N, prob.num_steps - 1, 2)
    assert dataset.s.shape == (Nx * L * N, prob.num_steps - 1, 2)
    assert dataset.k.shape == (Nx * L * N,)


if __name__ == "__main__":
    test_score_estimate()
    test_gen_from_state()
