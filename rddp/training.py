from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState

from rddp.utils import DiffusionDataset

Params = Any


@dataclass
class TrainingOptions:
    """Options for training a score network.

    The score network s_θ(x₀, U, σ) is trained to approximate the conditional
    noised score estimate ∇ log pₖ(U | x₀).

    Attributes:
        batch_size: The batch size for training.
        epochs: The number of training epochs.
        learning_rate: The learning rate for training.
    """

    batch_size: int
    epochs: int
    learning_rate: float


def create_train_state(
    network: nn.Module,
    state_shape: Tuple,
    action_shape: Tuple,
    options: TrainingOptions,
    rng: jax.random.PRNGKey,
) -> TrainState:
    """Initialize the parameters and optimizer.

    Args:
        network: The score network architecture.
        state_shape: The shape of the state x₀.
        action_shape: The shape of the control sequence U.
        options: The training options.
        rng: The random number generator.

    Returns:
        The initial training state.
    """
    params = network.init(
        rng,
        jnp.zeros((1, *state_shape)),  # x0
        jnp.zeros((1, *action_shape)),  # U
        jnp.zeros((1, 1)),  # sigma
    )
    optimizer = optax.adam(learning_rate=options.learning_rate)
    return TrainState.create(
        apply_fn=network.apply, params=params, tx=optimizer
    )


@jax.jit
def apply_model(
    state: TrainState, batch: DiffusionDataset
) -> Tuple[jnp.ndarray, Params]:
    """Compute the loss and gradients for a batch of data points.

    Args:
        state: The training state, including the network and parameters.
        batch: The batch of training data (x0, U, sigma, s).

    Returns:
        The training loss and parameter gradients.
    """

    def loss_fn(params: Params) -> jnp.ndarray:
        """Compute the loss for a batch of data points."""
        s = state.apply_fn(params, batch.x0, batch.U, batch.sigma)
        err = jnp.square(s - batch.s)

        # Weigh the error by the noise level, as recommended by Song et al.
        # Generative Modeling by Estimating Gradients of the Data Distribution,
        # NeurIPS 2019.
        err = jnp.einsum("ij,i...->i...", batch.sigma**2, err)

        return jnp.mean(err)

    loss, grad = jax.value_and_grad(loss_fn)(state.params)
    return loss, grad


def update_model(state: TrainState, grad: Params) -> TrainState:
    """Take a gradient descent step on the model parameters.

    Args:
        state: The training state.
        grad: The parameter gradients.

    Returns:
        The updated training state.
    """
    return state.apply_gradients(grads=grad)


def train_epoch(
    train_state: TrainState,
    dataset: DiffusionDataset,
    batch_size: int,
    rng: jax.random.PRNGKey,
) -> Tuple[TrainState, float]:
    """Perform an epoch of training on the given dataset.

    Args:
        train_state: The training state holding the parameters
        dataset: The training dataset.
        batch_size: The batch size.
        rng: The random number generator.

    Returns:
        The updated training state and the average training loss.
    """
    num_data_points = len(dataset.x0)
    steps_per_epoch = num_data_points // batch_size

    # Shuffle the dataset
    perms = jax.random.permutation(rng, num_data_points)
    perms = perms[: steps_per_epoch * batch_size]  # Truncate to full batches
    perms = perms.reshape((steps_per_epoch, batch_size))

    def apply_batch(train_state: TrainState, i: int):
        """Train the model on a single batch of data."""
        perm = perms[i]
        batch = jax.tree.map(lambda x: x[perm, ...], dataset)
        loss, grad = apply_model(train_state, batch)
        train_state = update_model(train_state, grad)
        return train_state, loss

    # Update the model on each batch
    train_state, losses = jax.lax.scan(
        apply_batch, train_state, jnp.arange(steps_per_epoch)
    )

    return train_state, jnp.mean(losses)


def train(
    network: nn.Module,  # TODO: make a custom score network type
    dataset: DiffusionDataset,
    options: TrainingOptions,
    seed: int = 0,
) -> Tuple[Params, dict]:
    """Train a score network on the given dataset.

    Args:
        network: The score network to train.
        dataset: The training dataset.
        options: The training options.
        seed: The random seed.

    Returns:
        The trained score network parameters and some training metrics.
    """
    rng = jax.random.PRNGKey(seed)

    # Shuffle the dataset
    num_data_points = len(dataset.x0)
    rng, shuffle_rng = jax.random.split(rng)
    perm = jax.random.permutation(shuffle_rng, num_data_points)
    dataset = jax.tree.map(lambda x: x[perm], dataset)

    # Split the dataset into training and validation sets
    num_train = num_data_points * 9 // 10
    train_data = jax.tree.map(lambda x: x[:num_train], dataset)
    val_data = jax.tree.map(lambda x: x[num_train:], dataset)

    # Put a cap on the validation set size to avoid OOM issues during training
    num_validation_points = len(val_data.x0)
    num_validation_points = min(num_validation_points, 10 * options.batch_size)
    val_data = jax.tree.map(lambda x: x[:num_validation_points], val_data)

    # Initialize the training state
    rng, init_rng = jax.random.split(rng)
    nx = dataset.x0.shape[-1:]  # TODO: support more generic shapes
    nu = dataset.U.shape[-2:]
    train_state = create_train_state(network, nx, nu, options, init_rng)

    # Train the model
    metrics = {"train_loss": [], "val_loss": []}
    for epoch in range(options.epochs):
        rng, epoch_rng = jax.random.split(rng)
        train_state, train_loss = train_epoch(
            train_state, train_data, options.batch_size, epoch_rng
        )
        val_loss, _ = apply_model(train_state, val_data)

        print(f"Epoch {epoch}, loss {train_loss:.4f}, val loss {val_loss:.4f}")
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)

    return train_state.params, metrics
