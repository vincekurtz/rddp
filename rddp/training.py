from pathlib import Path
from typing import Any, Tuple, Union
import time

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
        num_superbatches: The number of chunks to split the dataset into. Each
            superbatch is a collection of batches that are loaded into memory.
        epochs: The number of training epochs.
        learning_rate: The learning rate for training.
    """

    batch_size: int
    num_superbatches: int
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


def train_superbatch(
    train_state: TrainState,
    dataset: DiffusionDataset,
    options: TrainingOptions,
) -> Tuple[TrainState, dict]:
    """Train on a superbatch (large portion of the dataset).

    Args:
        train_state: The training state holding the parameters.
        dataset: The shuffled superbatch of training data, loaded on the GPU.
        options: The training options, like batch size etc.

    Returns:
        The updated training state and some training metrics.
    """
    # TODO: make these explicit args?
    superbatch_size = len(dataset.x0)
    num_batches = superbatch_size // options.batch_size
        
    def scan_fn(train_state: TrainState, batch: int):
        batch_data = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(
            x, batch * options.batch_size, options.batch_size), dataset)

        loss, grad = apply_model(train_state, batch_data)
        train_state = update_model(train_state, grad)
        return train_state, loss

    metrics = {}
    pre_train_time = time.time()
    train_state, loss = jax.lax.scan(scan_fn, train_state, jnp.arange(num_batches))
    metrics["train_time"] = time.time() - pre_train_time
    metrics["loss"] = jnp.mean(loss)

    return train_state, metrics


def train_epoch(
    train_state: TrainState,
    h5_dataset: HDF5DiffusionDataset,
    options: TrainingOptions, 
    rng: jax.random.PRNGKey,
) -> Tuple[TrainState, dict]:
    """Perform an epoch of training on the given dataset.

    Since loading a batch from disc is slow, but we can't fit the entire dataset
    into GPU memory, we load several "superbatches" at once. This allows
    us to train on the GPU without waiting too long for data to load.

    Args:
        train_state: The training state holding the parameters
        h5_dataset: The training dataset, loaded into CPU memory.
        options: The training options, like batch and superbatch sizes.
        rng: The random number generator.

    Returns:
        The updated training state and some training metrics
    """

    # Shuffle the dataset and split into superbatches
    num_data_points = len(h5_dataset)
    perm = jax.random.permutation(rng, num_data_points)
    perm = perm.reshape((options.num_superbatches, -1))
    superbatch_size = perm.shape[1]
    num_batches = superbatch_size // options.batch_size

    metrics = {"load_times": [], "train_times": [], "losses": []}
    for superbatch in perm:
        # Load data into GPU memory
        pre_load_time = time.time()
        dataset = h5_dataset[superbatch]
        metrics["load_times"].append(time.time() - pre_load_time)

        # Train on the superbatch
        def scan_fn(train_state: TrainState, batch: int):
            batch_data = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(
                x, batch * options.batch_size, options.batch_size), dataset)

            loss, grad = apply_model(train_state, batch_data)
            train_state = update_model(train_state, grad)
            return train_state, loss

        pre_train_time = time.time()
        train_state, losses = jax.lax.scan(scan_fn, train_state, jnp.arange(num_batches))

        metrics["train_times"].append(time.time() - pre_train_time)
        metrics["losses"].append(jnp.mean(losses))

    return train_state, metrics


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

    # Load the dataset from disc to CPU memory
    with h5py.File(dataset_file, "r") as f:
        h5_dataset = HDF5DiffusionDataset(f)

    # Compute some useful quantities and check sizes
    num_data_points = len(h5_dataset)
    assert num_data_points % options.num_superbatches == 0, (
        f"data points {num_data_points} not divisible by "
        f"number of superbatches {options.num_superbatches}."
    )
    superbatch_size = num_data_points // options.num_superbatches
    assert superbatch_size >= options.batch_size, (
        f"superbatch size {superbatch_size} is smaller than "
        f"batch size {options.batch_size}."
    )
    assert superbatch_size % options.batch_size == 0, (
        f"superbatch size {superbatch_size} not divisible by "
        f"batch size {options.batch_size}."
    )
    num_batches = superbatch_size // options.batch_size

    print("Training with: ")
    print(f"  {num_data_points} data points")
    print(f"  {options.batch_size} batch size")
    print(f"  {num_batches * options.num_superbatches} batches per epoch")
    print(f"  {options.num_superbatches} superbatches per epoch")
    print(f"  {options.epochs} epochs")

    # Initialize the training state
    rng, init_rng = jax.random.split(rng)
    nx = h5_dataset.x0.shape[-1:]  # TODO: support more generic shapes
    nu = h5_dataset.U.shape[-2:]
    train_state = create_train_state(network, nx, nu, options, init_rng)

    # Load the first superbatch into GPU memory
    # TODO: shuffle and handle case where num_superbatches > 1
    superbatch = h5_dataset[:] 

    metrics = {"train_loss": [], "val_loss": []}
    for epoch in range(options.epochs):
        # Shuffle the training data
        rng, epoch_rng = jax.random.split(rng)
        perm = jax.random.permutation(epoch_rng, superbatch_size)
        superbatch = jax.tree_map(lambda x: x[perm], superbatch)

        # Train on the superbatch
        train_state, train_metrics = train_superbatch(
            train_state, superbatch, options
        )

        # Print some training statistics
        print(f"Epoch {epoch + 1} / {options.epochs}, "
              f"train loss: {train_metrics['loss']:.4f}, "
              f"train time: {train_metrics['train_time']:.3f}s")

    return train_state.params, metrics
