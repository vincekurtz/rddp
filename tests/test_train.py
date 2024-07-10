from pathlib import Path

import jax
import jax.numpy as jnp

from rddp.architectures import DiffusionPolicyMLP
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.tasks.reach_avoid import ReachAvoid
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions


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

    # Create a temporary directory
    local_dir = Path("_test_training")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Generate a training dataset
    prob = ReachAvoidFixedX0(num_steps=5, start_state=jnp.array([0.1, -1.5]))
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=10,
        starting_noise_level=0.5,
        num_steps=4,
        step_size=0.01,
    )
    gen_options = DatasetGenerationOptions(
        noise_levels_per_file=5,
        temperature=0.001,
        num_initial_states=16,
        num_rollouts_per_data_point=8,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng, local_dir)

    # Train a score network
    options = TrainingOptions(
        batch_size=32,
        epochs=10,
        learning_rate=1e-3,
    )
    net = DiffusionPolicyMLP(layer_sizes=(16,) * 2)

    params, metrics = train(net, local_dir, options)
    assert metrics["train_loss"][-1] < metrics["train_loss"][0]

    # Remove the temporary directory
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_training()
