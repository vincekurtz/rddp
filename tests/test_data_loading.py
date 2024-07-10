from pathlib import Path

import jax
import jax.numpy as jnp

from rddp.data_loading import TorchDiffusionDataLoader, TorchDiffusionDataset
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.tasks.reach_avoid import ReachAvoid
from rddp.utils import AnnealedLangevinOptions


def test_data_loader() -> None:
    """Test the torch data loader."""
    # Create a temporary directory
    local_dir = Path("_test_data_loader")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create a generator
    prob = ReachAvoid(num_steps=10)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=30,
        starting_noise_level=0.1,
        num_steps=8,
        step_size=0.1,
    )
    gen_options = DatasetGenerationOptions(
        noise_levels_per_file=10,
        temperature=0.01,
        num_initial_states=3,
        num_rollouts_per_data_point=16,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate and save the dataset
    rng = jax.random.PRNGKey(0)
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng, local_dir)

    # Create a torch dataset from the saved data
    torch_dataset = TorchDiffusionDataset(local_dir)

    assert (
        len(torch_dataset)
        == langevin_options.num_noise_levels
        * langevin_options.num_steps
        * gen_options.num_initial_states
    )

    loader = TorchDiffusionDataLoader(
        torch_dataset, batch_size=32, shuffle=True
    )
    data = next(iter(loader))

    assert isinstance(data, dict)
    assert isinstance(data["x0"], jnp.ndarray)
    assert data["x0"].shape == (32, 2)
    assert data["U"].shape == (32, 9, 2)
    assert data["s"].shape == (32, 9, 2)
    assert data["sigma"].shape == (32, 1)

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_data_loader()
