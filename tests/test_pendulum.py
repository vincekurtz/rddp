import jax

from rddp.envs.pendulum import PendulumSwingupEnv


def test_step_reset() -> None:
    """Test the basic step and reset functions."""
    rng = jax.random.PRNGKey(0)
    env = PendulumSwingupEnv()

    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    assert state.pipeline_state.q.shape == (1,)
    assert state.pipeline_state.qd.shape == (1,)

    action = jax.random.uniform(rng, (1,), minval=-1.0, maxval=1.0)
    state = env.step(state, action)
    assert state.pipeline_state.q.shape == (1,)
    assert state.pipeline_state.qd.shape == (1,)


if __name__ == "__main__":
    test_step_reset()
