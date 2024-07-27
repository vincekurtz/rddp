import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax.envs.base import State

from rddp.envs.reach_avoid import ReachAvoidEnv


def test_plot() -> None:
    """Test plotting the reach-avoid scenario."""
    env = ReachAvoidEnv(num_steps=10)
    env.plot_scenario()
    if __name__ == "__main__":
        # Only show the plot if this script is run directly, not in pytest.
        plt.show()


def test_step_reset() -> None:
    """Test the basic step and reset functions."""
    rng = jax.random.PRNGKey(0)
    env = ReachAvoidEnv(num_steps=10)

    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    assert isinstance(state, State)
    start_pos = state.pipeline_state.q
    assert start_pos.shape == (2,)
    assert state.info["step"] == 0

    vel = jnp.array([0.8, 0.3])
    state = env.step(state, vel)
    assert jnp.allclose(state.pipeline_state.q - start_pos, vel)
    assert state.info["step"] == 1


if __name__ == "__main__":
    test_plot()
    test_step_reset()
