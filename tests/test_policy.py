from pathlib import Path

import jax
import jax.numpy as jnp

from rddp.architectures import ScoreMLP
from rddp.envs.reach_avoid import ReachAvoidEnv
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.ocp import OptimalControlProblem
from rddp.policy import DiffusionPolicy
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions


def test_policy() -> None:
    """Test the DiffusionPolicy helper object."""
    rng = jax.random.PRNGKey(0)

    local_dir = Path("_test_policy")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Quickly train a little policy
    prob = OptimalControlProblem(ReachAvoidEnv(num_steps=5), num_steps=5)
    langevin_options = AnnealedLangevinOptions(
        denoising_steps=100,
        starting_noise_level=0.8,
        step_size=0.01,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=16,
        num_rollouts_per_data_point=8,
        save_every=100,
        print_every=10,
        save_path=local_dir,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)
    rng, gen_rng = jax.random.split(rng)
    generator.generate(gen_rng)
    assert (local_dir / "dataset.h5").exists()

    train_options = TrainingOptions(
        batch_size=100,
        num_superbatches=1,
        epochs=2,
        learning_rate=1e-3,
    )
    net = ScoreMLP(layer_sizes=(32,) * 3)
    params, _ = train(net, local_dir / "dataset.h5", train_options)

    # Create a policy object
    action_shape = (prob.num_steps - 1, prob.env.action_size)
    policy = DiffusionPolicy(net, params, langevin_options, action_shape)

    # Test the policy
    y0 = jnp.zeros(2)
    rng, policy_rng = jax.random.split(rng)
    U = policy.apply(y0, policy_rng)
    assert U.shape == action_shape

    # Save the policy to a file
    policy.save(local_dir / "policy.pkl")

    # load the policy from a file and check that it is the same
    del policy
    new_policy = DiffusionPolicy.load(local_dir / "policy.pkl")
    new_U = new_policy.apply(y0, policy_rng)

    assert jnp.allclose(U, new_U)

    # Clean up
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_policy()
