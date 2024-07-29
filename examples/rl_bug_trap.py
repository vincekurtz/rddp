import functools
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax import envs
from brax.training.agents.ppo import train as ppo

from rddp.envs.bug_trap import BugTrapEnv

##
#
# Solve the bug trap example with Reinforcement Learning (PPO).
#
##


# Global planning horizon definition
HORIZON = 20

# Set up the environment
envs.register_environment("bug_trap", lambda: BugTrapEnv(num_steps=HORIZON))
env = envs.get_environment("bug_trap")

# Set up training
train_fn = functools.partial(
    ppo.train,
    num_timesteps=10_000_000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=HORIZON,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=2048,
    batch_size=1024,
    clipping_epsilon=0.2,
    seed=0,
)

times = [datetime.now()]


def progress_fn(step: int, metrics: dict) -> None:
    """Callback function to print out training progress."""
    times.append(datetime.now())
    print(
        f'  step={step}, reward={metrics["eval/episode_reward"]}, '
        f'time={times[-1]}'
    )


# Train the agent
print("Training PPO policy")
make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=progress_fn
)

print(f"Time to jit: {times[1] - times[0]}")
print(f"Time to train: {times[-1] - times[1]}")

# set up the policy
inference_fn = make_inference_fn(params)
jit_policy = jax.jit(inference_fn)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Roll out the policy
print("\nRolling out policy")
rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)
state = jit_reset(reset_rng)

trajectory = [state.obs]
cost = 0.0
for _ in range(HORIZON):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_policy(state.obs, act_rng)
    state = jit_step(state, ctrl)
    trajectory.append(state.obs)
    cost -= state.reward
print("Trajectory cost:", cost)

# Plot the trajectory
env.plot_scenario()
trajectory = jnp.stack(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], "o-")
plt.show()
