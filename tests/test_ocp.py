import jax
import jax.numpy as jnp

from rddp.envs.reach_avoid import ReachAvoidEnv
from rddp.ocp import OptimalControlProblem


def test_rollout() -> None:
    """Test that we can roll out a control tape and compute the total cost."""
    rng = jax.random.PRNGKey(0)

    env = ReachAvoidEnv(num_steps=10)
    ocp = OptimalControlProblem(env, num_steps=10)

    rng, x0_rng, u_rng = jax.random.split(rng, 3)
    x0 = env.reset(x0_rng)
    control_tape = jax.random.uniform(u_rng, (ocp.num_steps, 2))

    # Roll out with the helper
    total_cost, state_trajectory = ocp.rollout(x0, control_tape)
    assert state_trajectory.pipeline_state.q.shape == (ocp.num_steps + 1, 2)
    assert jnp.all(state_trajectory.done[0:-1] == 0.0)
    assert state_trajectory.done[-1] == 1.0

    # Roll out manually
    x = x0
    total_cost_manual = 0.0
    for i in range(ocp.num_steps):
        assert jnp.all(x.done == 0.0)
        u = control_tape[i]
        x = env.step(x, u)
        total_cost_manual -= x.reward
    assert jnp.all(x.done == 1.0)

    assert jnp.allclose(total_cost, total_cost_manual)
    assert jnp.allclose(
        state_trajectory.pipeline_state.q[-1], x.pipeline_state.q
    )


if __name__ == "__main__":
    test_rollout()
