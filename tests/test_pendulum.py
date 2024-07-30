import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.envs.pendulum import PendulumEnv


def test_plot() -> None:
    """Test the basic dynamics of the double integrator."""
    env = PendulumEnv()
    assert env.action_size == 1
    assert env.observation_size == 3
    env.plot_scenario()
    if __name__ == "__main__":
        # Only show the plot if this script is run directly, not in pytest.
        plt.show()


def test_step_reset() -> None:
    """Test the basic step and reset functions."""
    rng = jax.random.PRNGKey(0)
    env = PendulumEnv()

    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    assert state.pipeline_state.q.shape == (1,)
    assert state.pipeline_state.qd.shape == (1,)
    assert state.obs.shape == (3,)

    u = jnp.array([0.8])
    new_state = env.step(state, u)
    assert new_state.pipeline_state.q.shape == (1,)
    assert new_state.pipeline_state.qd.shape == (1,)


if __name__ == "__main__":
    test_plot()
    test_step_reset()
