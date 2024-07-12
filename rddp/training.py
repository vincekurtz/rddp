from pathlib import Path
from typing import Any, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState

from rddp.utils import DiffusionDataset, HDF5DiffusionDataset

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


@jax.jit
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
    dataset_file: Union[str, Path],
    options: TrainingOptions,
    seed: int = 0,
) -> Tuple[Params, dict]:
    """Train a score network on the given dataset.

    Args:
        network: The score network to train.
        dataset_file: Path to the hdf5 file containing the dataset.
        options: The training options.
        seed: The random seed.

    Returns:
        The trained score network parameters and some training metrics.
    """
    rng = jax.random.PRNGKey(seed)
    dataset_file = Path(dataset_file)

    print(f"Loading dataset from {dataset_file}...")

    with h5py.File(dataset_file, "r") as f:
        # Load the data into an intermediate hdf5 representation. This keeps
        # everything on disc, but allows us to read into GPU memory in chunks.
        h5_dataset = HDF5DiffusionDataset(f)
        num_data_points = len(h5_dataset)
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, num_data_points)

        # Split the dataset into training and validation sets
        print("Creating training and validation sets...")
        num_val = num_data_points // 10
        num_val = min(num_val, options.batch_size)  # Cap to avoid OOM issues
        num_train = num_data_points - num_val
        num_batches = num_train // options.batch_size
        train_idx = perm[:num_train]
        val_idx = perm[num_train:]

        # Load the validation data into memory
        # N.B. we must sort the indices to avoid HDF5 chunking issues
        print(f"Loading {num_val} validation data points...")
        val_data = h5_dataset[np.sort(val_idx)]

        # Initialize the training state
        print("Initializing training state...")
        rng, init_rng = jax.random.split(rng)
        nx = h5_dataset.x0.shape[-1:]  # TODO: support more generic shapes
        nu = h5_dataset.U.shape[-2:]
        train_state = create_train_state(network, nx, nu, options, init_rng)

        print("Getting initial validation loss...")
        val_loss, _ = apply_model(train_state, val_data)
        print("Initial val loss:", val_loss)

        metrics = {"train_loss": [], "val_loss": []}
        for epoch in range(options.epochs):
            # Shuffle the training dataset
            rng, epoch_rng = jax.random.split(rng)
            batch_perms = jax.random.permutation(epoch_rng, num_train)
            batch_perms = batch_perms[: num_batches * options.batch_size]
            batch_perms = batch_perms.reshape((num_batches, options.batch_size))

            # Train the model on each batch
            for batch in batch_perms:
                batch_data = h5_dataset[np.sort(train_idx[batch])]
                loss, grad = apply_model(train_state, batch_data)
                train_state = update_model(train_state, grad)

            # Print some metrics
            val_loss, _ = apply_model(train_state, val_data)
            metrics["train_loss"].append(loss)
            metrics["val_loss"].append(val_loss)
            print(
                f"Epoch {epoch + 1}: "
                f"train loss {loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )

    return train_state.params, metrics
