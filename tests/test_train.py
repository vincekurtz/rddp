import jax
import jax.numpy as jnp

from rddp.architectures import DiffusionPolicyMLP
from rddp.data_generation import DatasetGenerationOptions, DatasetGenerator
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

    # Generate a training dataset
    prob = ReachAvoidFixedX0(num_steps=5, start_state=jnp.array([0.1, -1.5]))
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=100,
        starting_noise_level=0.5,
        num_steps=10,
        step_size=0.01,
    )
    gen_options = DatasetGenerationOptions(
        temperature=0.001,
        num_initial_states=16,
        num_rollouts_per_data_point=8,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)
    rng, gen_rng = jax.random.split(rng)
    dataset = generator.generate(gen_rng)

    # Train a score network
    options = TrainingOptions(
        batch_size=128,
        epochs=4,
        learning_rate=1e-3,
    )
    net = DiffusionPolicyMLP(layer_sizes=(32,) * 3)

    params, metrics = train(net, dataset, options)
    assert metrics["train_loss"][-1] < metrics["train_loss"][0]
    assert metrics["val_loss"][-1] < metrics["val_loss"][0]

    test_idx = 129
    score_estimate = net.apply(
        params,
        dataset.x0[test_idx],
        dataset.U[test_idx],
        dataset.sigma[test_idx],
    )
    assert score_estimate.shape == dataset.s[test_idx].shape


if __name__ == "__main__":
    test_training()
