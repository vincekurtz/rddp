import jax
import jax.numpy as jnp

from rddp.envs.reach_avoid import ReachAvoidEnv
from rddp.ocp import OptimalControlProblem


def test_rollout() -> None:
    """Test that we can roll out a control tape and compute the total cost."""
    rng = jax.random.PRNGKey(0)

    env = ReachAvoidEnv()
    ocp = OptimalControlProblem(env, num_steps=10)

    rng, x0_rng, u_rng = jax.random.split(rng, 3)
    x0 = env.reset(x0_rng)
    control_tape = jax.random.uniform(u_rng, (ocp.num_steps, 2))

    # Roll out with the helper
    total_cost, state_trajectory = ocp.rollout(x0, control_tape)

    # Roll out manually
    x = x0
    total_cost_manual = 0.0
    for i in range(ocp.num_steps):
        u = control_tape[i]
        x = env.step(x, u)
        total_cost_manual -= x.reward

    assert jnp.allclose(total_cost, total_cost_manual)
    assert jnp.allclose(
        state_trajectory.pipeline_state.q[-1], x.pipeline_state.q
    )


if __name__ == "__main__":
    test_rollout()
