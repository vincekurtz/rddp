import time
from typing import Tuple

import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv
from brax.envs.inverted_pendulum import InvertedPendulum

from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.ocp import OptimalControlProblem
from rddp.utils import AnnealedLangevinOptions


def simulate(
    env: PipelineEnv, num_timesteps: int, num_parallel_envs: int
) -> float:
    """Just simulate a brax env in parallel and report the time it took.

    Args:
        env: The environment to simulate.
        num_timesteps: The number of timesteps to simulate.
        num_parallel_envs: The number of parallel environments to simulate.

    Returns:
        The time it took to run the simulations (including jitting).
    """
    rng = jax.random.PRNGKey(0)

    def _rollout(rng: jax.random.PRNGKey):
        """Roll out a fixed environment for a given number of timesteps."""
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(rng=reset_rng)

        def _scan_fn(carry: Tuple, t: int):
            state, rng = carry
            rng, action_rng = jax.random.split(rng)
            action = jax.random.normal(action_rng, (env.action_size,))
            state = env.step(state, action)
            return (state, rng), None

        (state, rng), _ = jax.lax.scan(
            _scan_fn, (state, rng), jnp.arange(num_timesteps)
        )

        return state, rng

    rng, env_rng = jax.random.split(rng)
    env_rng = jax.random.split(env_rng, num_parallel_envs)

    st = time.time()
    states, _ = jax.jit(jax.vmap(_rollout))(env_rng)
    jax.block_until_ready(states.obs)
    gen_time = time.time() - st
    print(
        f"Simulated {num_timesteps} steps across {num_parallel_envs} "
        f"parallel envs in {gen_time:.2f} seconds"
    )
    return gen_time


def test_generation_speed() -> None:
    """Run dataset generation and compare to straight-up simulation."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/speed_test/"

    env = InvertedPendulum()
    prob = OptimalControlProblem(env, num_steps=15)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=20,
        starting_noise_level=0.1,
        step_size=1.0,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=16,
        num_rollouts_per_data_point=8,
        save_every=20,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    rng, gen_rng = jax.random.split(rng)
    st = time.time()
    generator.generate(gen_rng)
    gen_time = time.time() - st

    num_timesteps = prob.num_steps * langevin_options.num_noise_levels
    num_parallel_envs = (
        gen_options.num_initial_states * gen_options.num_rollouts_per_data_point
    )

    print(
        f"Generated {num_timesteps} steps across {num_parallel_envs} "
        f"parallel envs in {gen_time:.2f} seconds"
    )

    sim_time = simulate(env, num_timesteps, num_parallel_envs)

    # The straightforward sim is usually faster (especially with jitting),
    # but it shouldn't be wildly faster.
    assert gen_time < 1.5 * sim_time


if __name__ == "__main__":
    test_generation_speed()
