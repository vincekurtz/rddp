import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rddp.data_generation import (
    AnnealedLangevinOptions,
    DatasetGenerationOptions,
    DatasetGenerator,
)
from rddp.tasks.base import OptimalControlProblem
from rddp.tasks.reach_avoid import ReachAvoid


class ReachAvoidFixedX0(ReachAvoid):
    """A reach-avoid problem with a fixed initial state."""

    def __init__(self, num_steps: int, start_state: jnp.ndarray):
        """Initialize the reach-avoid problem.

        Args:
            num_steps: The number of time steps T.
            start_state: The initial state x0.
        """
        super().__init__(num_steps)
        self.x0 = start_state

    def sample_initial_state(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample the initial state x₀."""
        return self.x0


def solve_with_gradient_descent(
    prob: OptimalControlProblem, x0: jnp.ndarray
) -> jnp.ndarray:
    """Solve the optimal control problem using simple gradient descent.

    Args:
        prob: The optimal control problem.
        x0: The initial state.

    Returns:
        The optimal state trajectory.
    """
    cost_and_grad = jax.jit(
        jax.value_and_grad(lambda us: prob.total_cost(us, x0))
    )
    U = jnp.zeros((prob.num_steps - 1, prob.sys.action_shape[0]))
    J, grad = cost_and_grad(U)

    for i in range(5000):
        J, grad = cost_and_grad(U)
        U -= 1e-2 * grad

        if i % 1000 == 0:
            print(f"Step {i}, cost {J}, grad {jnp.linalg.norm(grad)}")

    return prob.sys.rollout(U, x0)


def plot_dataset() -> None:
    """Generate some data and make plots of it."""
    rng = jax.random.PRNGKey(0)
    x0 = jnp.array([0.1, -1.5])

    # Problem setup
    prob = ReachAvoidFixedX0(num_steps=20, start_state=x0)
    langevin_options = AnnealedLangevinOptions(
        temperature=0.01,
        num_noise_levels=100,
        starting_noise_level=1.0,
        noise_decay_rate=0.95,
    )
    gen_options = DatasetGenerationOptions(
        num_initial_states=3,
        num_data_points_per_initial_state=4,
        num_rollouts_per_data_point=64,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate some data
    rng, gen_rng = jax.random.split(rng)
    dataset = generator.generate(gen_rng)

    # Make some plots
    gamma = langevin_options.noise_decay_rate
    sigma_L = langevin_options.starting_noise_level
    L = langevin_options.num_noise_levels

    # Plot samples at certain noise levels
    fig, ax = plt.subplots(1, 5)

    for i, k in enumerate([0, 25, 50, 75, 99]):
        plt.sca(ax[i])
        prob.plot_scenario()
        sigma = sigma_L * gamma ** (L - k - 1)
        ax[i].set_title(f"k={k}, σₖ={sigma:.4f}")
        idxs = jnp.where(dataset.k == k)
        Us = dataset.U[idxs]
        x0s = dataset.x0[idxs]
        Xs = jax.vmap(prob.sys.rollout)(Us, x0s)
        px = Xs[:, :, 0].T
        py = Xs[:, :, 1].T
        ax[i].plot(px, py, "o-", color="blue", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    plot_dataset()

    # prob = ReachAvoid(num_steps=20)
    # prob.plot_scenario()
    # x0 = jnp.array([0.1, -1.5])
    # X = solve_with_gradient_descent(prob, x0)
    # plt.plot(X[:, 0], X[:, 1], "o-")
    # plt.show()
