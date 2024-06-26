import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from rddp.architectures import DiffusionPolicyMLP
from rddp.data_generation import (
    AnnealedLangevinOptions,
    DatasetGenerationOptions,
    DatasetGenerator,
)
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


def solve_with_gradient_descent() -> None:
    """Solve the optimal control problem using simple gradient descent."""
    prob = ReachAvoid(num_steps=20)
    x0 = jnp.array([0.1, -1.5])

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

    X = prob.sys.rollout(U, x0)

    prob.plot_scenario()
    plt.plot(X[:, 0], X[:, 1], "o-")
    plt.show()


def generate_dataset(save: bool = False) -> None:
    """Generate some data and make plots of it."""
    rng = jax.random.PRNGKey(0)
    x0 = jnp.array([-0.1, -1.5])

    # Problem setup
    prob = ReachAvoidFixedX0(num_steps=20, start_state=x0)
    langevin_options = AnnealedLangevinOptions(
        temperature=0.001,
        num_noise_levels=100,
        starting_noise_level=1.0,
        noise_decay_rate=0.95,
    )
    gen_options = DatasetGenerationOptions(
        num_initial_states=5,
        num_data_points_per_initial_state=16,
        num_rollouts_per_data_point=64,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    # Generate some data
    rng, gen_rng = jax.random.split(rng)
    dataset = generator.generate(gen_rng)

    # Save the data if requested
    fname = "/tmp/reach_avoid_dataset.pkl"
    if save:
        generator.save_dataset(dataset, fname)
        print(f"Saved dataset to {fname}")

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

    # Plot cost at each iteration
    plt.figure()

    jit_cost = jax.jit(jax.vmap(prob.total_cost))
    for k in range(L):
        iter = L - k
        idxs = jnp.where(dataset.k == k)
        Us = dataset.U[idxs]
        x0s = dataset.x0[idxs]
        costs = jit_cost(Us, x0s)
        plt.scatter(jnp.ones_like(costs) * iter, costs, color="blue", alpha=0.5)
    plt.xlabel("Iteration (L - k)")
    plt.ylabel("Cost J(U, x₀)")
    plt.yscale("log")

    plt.show()


def fit_score_model() -> None:
    """Fit a simple scor32e model to the generated data."""
    rng = jax.random.PRNGKey(0)

    # Load training data from a file (must run generate_dataset first)
    fname = "/tmp/reach_avoid_dataset.pkl"
    with open(fname, "rb") as f:
        data = pickle.load(f)
    dataset = data["dataset"]
    dataset = dataset.replace(k=dataset.k.reshape(-1, 1))  # TODO: fix earlier
    options = data["langevin_options"]

    # Split the data into training and validation sets
    rng, split_rng = jax.random.split(rng)
    idxs = jax.random.permutation(split_rng, len(dataset.x0))
    train_idxs = idxs[: int(0.8 * len(idxs))]
    val_idxs = idxs[int(0.8 * len(idxs)) :]

    train_dataset = jax.tree.map(lambda x: x[train_idxs], dataset)
    val_dataset = jax.tree.map(lambda x: x[val_idxs], dataset)

    # Initialize the score model
    net = DiffusionPolicyMLP(layer_sizes=[128, 128])
    dummy_x0 = jnp.zeros((2,))
    dummy_U = jnp.zeros((19, 2))
    dummy_k = jnp.array([10])
    rng, params_rng = jax.random.split(rng)
    params = net.init(params_rng, dummy_x0, dummy_U, dummy_k)

    # Learning parameters
    iterations = 2000
    batch_size = 256
    learning_rate = 1e-3

    # Training loop
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(params, x0, U, k, s):  # noqa: ANN001, N803 (ingore annotations)
        """Loss function for score model training."""
        s_pred = net.apply(params, x0, U, k)
        return jnp.mean(jnp.square(s_pred - s))

    loss_and_grad = jax.value_and_grad(loss_fn)
    jit_loss = jax.jit(loss_fn)

    @jax.jit
    def train_step(params, opt_state, rng):  # noqa: ANN001 (ingore annotations)
        """Perform a single SGD step."""
        # Sample a batch
        idxs = jax.random.randint(rng, (batch_size,), 0, len(train_dataset.x0))
        x0 = train_dataset.x0[idxs]
        U = train_dataset.U[idxs]
        k = train_dataset.k[idxs]
        s = train_dataset.s[idxs]

        # Compute the loss and gradients
        loss, grad = loss_and_grad(params, x0, U, k, s)

        # Update the parameters
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    for iter in range(iterations):
        rng, batch_rng = jax.random.split(rng)
        params, opt_state, loss = train_step(params, opt_state, batch_rng)

        if iter % 500 == 0 or iter == iterations - 1:
            val_loss = jit_loss(
                params,
                val_dataset.x0,
                val_dataset.U,
                val_dataset.k,
                val_dataset.s,
            )
            print(f"Step {iter}, loss {loss:.4f}, val loss {val_loss:.4f}")

    # Save the trained model and parameters
    fname = "/tmp/reach_avoid_score_model.pkl"
    with open(fname, "wb") as f:
        pickle.dump({"params": params, "net": net, "options": options}, f)


def deploy_trained_model() -> None:
    """Use the trained model to generate optimal actions."""
    rng = jax.random.PRNGKey(0)

    # Set up the system
    x0 = jnp.array([-0.1, -1.5])
    prob = ReachAvoidFixedX0(num_steps=20, start_state=x0)

    # Load the trained score network
    with open("/tmp/reach_avoid_score_model.pkl", "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    net = data["net"]
    options = data["options"]

    # Do annealed langevin sampling
    L = options.num_noise_levels
    sigma = options.starting_noise_level
    eps = 0.001

    rng, init_rng = jax.random.split(rng)
    U = sigma * jax.random.normal(init_rng, (prob.num_steps - 1, 2))

    for k in range(L - 1, -1, -1):
        cost = prob.total_cost(U, x0)
        print(f"k = {k}, cost = {cost}")

        for _ in range(20):
            # Langevin sampling at this noise level
            rng, noise_rng = jax.random.split(rng)
            z = jax.random.normal(noise_rng, U.shape)
            score = net.apply(params, x0, U, jnp.array([k]))
            U = U + eps * score + jnp.sqrt(2 * eps * sigma**2) * z

        # Plot stuff every now and then
        if k % 20 == 0:
            plt.figure()
            X = prob.sys.rollout(U, x0)
            prob.plot_scenario()
            plt.plot(X[:, 0], X[:, 1], "o-")
            plt.title(f"k={k}, σₖ={sigma:.4f}")

        # Anneal the noise
        sigma *= options.noise_decay_rate

    plt.show()


if __name__ == "__main__":
    # solve_with_gradient_descent()
    # generate_dataset(save=True)
    # fit_score_model()
    deploy_trained_model()
