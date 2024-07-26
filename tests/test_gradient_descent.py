import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.envs.reach_avoid import ReachAvoidEnv
from rddp.gradient_descent import solve
from rddp.ocp import OptimalControlProblem


def test_gradient_descent() -> None:
    """Test solving the reach-avoid problem using gradient descent."""
    rng = jax.random.PRNGKey(2)
    prob = OptimalControlProblem(ReachAvoidEnv(), num_steps=10)

    rng, reset_rng = jax.random.split(rng)
    x0 = prob.env.reset(reset_rng)

    U, J, grad = solve(prob, x0, max_iter=1000)
    _, X = prob.rollout(x0, U)

    assert jnp.linalg.norm(grad) < 1e-4

    positions = X.pipeline_state.q
    target_position = prob.env.target_position

    assert jnp.allclose(positions[0], x0.pipeline_state.q)
    assert jnp.linalg.norm(positions[-1] - target_position) < 0.3

    if __name__ == "__main__":
        # Only make plots for a manual run
        prob.env.plot_scenario()
        plt.plot(positions[:, 0], positions[:, 1], "o-")
        plt.show()


if __name__ == "__main__":
    test_gradient_descent()
