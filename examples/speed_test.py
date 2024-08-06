##
#
# Debug dataset generation times
#
##

import time
from typing import Tuple

import jax
import jax.numpy as jnp
from brax.envs.inverted_pendulum import InvertedPendulum

from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.ocp import OptimalControlProblem
from rddp.utils import AnnealedLangevinOptions


def generate() -> None:
    """Run dataset generation."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/speed_test/"

    prob = OptimalControlProblem(InvertedPendulum(), num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=30,
        starting_noise_level=0.1,
        num_steps=1,
        step_size=1.0,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=16,
        num_rollouts_per_data_point=8,
        save_every=1,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    rng, gen_rng = jax.random.split(rng)
    st = time.time()
    # generator.generate_and_save(gen_rng)
    generator.generate(gen_rng)
    gen_time = time.time() - st

    num_timesteps = (
        prob.num_steps
        * langevin_options.num_noise_levels
        * langevin_options.num_steps
    )
    num_parallel_envs = (
        gen_options.num_initial_states * gen_options.num_rollouts_per_data_point
    )

    print(
        f"Simulated {num_timesteps} steps across {num_parallel_envs} "
        f"parallel envs in {gen_time:.2f} seconds"
    )


def manual_sim() -> None:
    """Manually just simulate, no Langevin sampling or anything."""
    num_timesteps = 600
    num_parallel_envs = 128

    env = InvertedPendulum()

    def simulate(rng: jax.random.PRNGKey):
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

    rng = jax.random.PRNGKey(0)
    rng, env_rng = jax.random.split(rng)
    env_rng = jax.random.split(env_rng, num_parallel_envs)

    st = time.time()
    states, _ = jax.jit(jax.vmap(simulate))(env_rng)
    gen_time = time.time() - st
    print(
        f"Simulated {num_timesteps} steps across {num_parallel_envs} "
        f"parallel envs in {gen_time:.2f} seconds"
    )


def manual_rollouts() -> None:
    """Manually perform Langevin sampling."""
    rng = jax.random.PRNGKey(0)

    prob = OptimalControlProblem(InvertedPendulum(), num_steps=20)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=30,
        starting_noise_level=0.1,
        num_steps=1,
        step_size=1.0,
        noise_injection_level=1.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=16,
        num_rollouts_per_data_point=8,
        save_every=1,
        save_path="/tmp/debug_manual_rollouts",
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    rng, x_rng, u_rng = jax.random.split(rng, 3)
    x_rng = jax.random.split(x_rng, gen_options.num_initial_states)
    x0 = jax.vmap(jax.jit(prob.env.reset))(x_rng)
    U = jax.random.normal(
        u_rng,
        (
            gen_options.num_initial_states,
            prob.num_steps - 1,
            prob.env.action_size,
        ),
    )

    jit_score = jax.jit(
        jax.vmap(generator.estimate_noised_score, in_axes=(0, 0, None, None))
    )

    st = time.time()
    for _ in range(
        langevin_options.num_noise_levels * langevin_options.num_steps
    ):
        rng, score_rng, noise_rng = jax.random.split(rng, 3)
        s = jit_score(x0, U, 0.1, score_rng)
        noise = jax.random.normal(noise_rng, U.shape)
        U += 0.01 * s + jnp.sqrt(0.02) * noise
        jax.block_until_ready(U)
        print(time.time() - st)
    total_time = time.time() - st

    num_timesteps = (
        prob.num_steps
        * langevin_options.num_noise_levels
        * langevin_options.num_steps
    )
    num_parallel_envs = (
        gen_options.num_initial_states * gen_options.num_rollouts_per_data_point
    )
    print(
        f"Simulated {num_timesteps} steps across {num_parallel_envs} "
        f"parallel envs in {total_time:.2f} seconds"
    )


if __name__ == "__main__":
    generate()
    # manual_sim()
    # manual_rollouts()
