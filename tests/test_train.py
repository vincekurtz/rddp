from pathlib import Path

import h5py
import jax
import jax.numpy as jnp

from rddp.architectures import ScoreMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.tasks.reach_avoid import ReachAvoid
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions, HDF5DiffusionDataset


class ReachAvoidFixedX0(ReachAvoid):
    """A reach-avoid problem with a fixed initial state."""

    def __init__(self, num_steps: int, start_state: jnp.ndarray):
        """Initialize the reach-avoid problem.

        Args:
            num_steps: The number of time steps T.
            start_state: The initial state x0.
        """
        super().__init__(num_steps)
        self.x0 = start_state

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample the initial state x₀."""
        return self.x0


def test_training() -> None:
    """Test the main training loop."""
    rng = jax.random.PRNGKey(0)

    # Make a temporary directory for the dataset
    local_dir = Path("_test_training")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Generate a training dataset
    prob = ReachAvoidFixedX0(num_steps=5, start_state=jnp.array([0.1, -1.5]))
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=100,
        starting_noise_level=0.5,
        num_steps=10,
        step_size=0.01,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
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
        num_superbatches=4,
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
    test_training()
