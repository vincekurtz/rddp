import functools
import time
from pathlib import Path
from typing import Any, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState

from rddp.utils import DiffusionDataset, HDF5DiffusionDataset

Params = Any


@dataclass
class TrainingOptions:
    """Options for training a score network.

    The score network s_θ(y₀, U, σ) is trained to approximate the conditional
    noised score estimate ∇ log pₖ(U | y₀).

    Attributes:
        batch_size: The batch size for training.
        num_superbatches: The number of chunks to split the dataset into. Each
            superbatch is a collection of batches that are loaded into memory.
        epochs: The number of training epochs.
        learning_rate: The learning rate for training.
        print_every: The frequency at which to print training statistics.
    """

    batch_size: int
    num_superbatches: int
    epochs: int
    learning_rate: float
    print_every: int = 1


def create_train_state(
    network: nn.Module,
    obs_shape: Tuple,
    action_shape: Tuple,
    options: TrainingOptions,
    rng: jax.random.PRNGKey,
) -> TrainState:
    """Initialize the parameters and optimizer.

    Args:
        network: The score network architecture.
        obs_shape: The shape of the observation y₀.
        action_shape: The shape of the control sequence U.
        options: The training options.
        rng: The random number generator.

    Returns:
        The initial training state.
    """
    params = network.init(
        rng,
        jnp.zeros((1, *obs_shape)),  # y0
        jnp.zeros((1, *action_shape)),  # U
        jnp.zeros((1, 1)),  # sigma
    )
    optimizer = optax.adam(learning_rate=options.learning_rate)
    return TrainState.create(
        apply_fn=network.apply, params=params, tx=optimizer
    )


def apply_model(
    state: TrainState, batch: DiffusionDataset
) -> Tuple[jnp.ndarray, Params]:
    """Compute the loss and gradients for a batch of data points.

    Args:
        state: The training state, including the network and parameters.
        batch: The batch of training data (y0, U, sigma, s).

    Returns:
        The training loss and parameter gradients.
    """

    def loss_fn(params: Params) -> jnp.ndarray:
        """Compute the loss for a batch of data points."""
        s = state.apply_fn(params, batch.Y[0], batch.U, batch.sigma)
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


@functools.partial(jax.jit, static_argnums=(2,), donate_argnums=(0,))
def train_superbatch(
    train_state: TrainState,
    dataset: DiffusionDataset,
    options: TrainingOptions,
) -> Tuple[TrainState, jnp.ndarray]:
    """Train on a superbatch (large portion of the dataset).

    Args:
        train_state: The training state holding the parameters.
        dataset: The shuffled superbatch of training data, loaded on the GPU.
        options: The training options, like batch size etc.

    Returns:
        The updated training state and the latest training loss
    """
    superbatch_size = len(dataset.Y)
    num_batches = superbatch_size // options.batch_size

    def scan_fn(carry: Tuple[TrainState, jnp.ndarray], batch: int):
        train_state, _ = carry
        batch_data = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, batch * options.batch_size, options.batch_size
            ),
            dataset,
        )

        loss, grad = apply_model(train_state, batch_data)
        train_state = update_model(train_state, grad)
        return (train_state, loss), None

    (train_state, loss), _ = jax.lax.scan(
        scan_fn, (train_state, 0.0), jnp.arange(num_batches)
    )

    return train_state, loss


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
    ny = h5_dataset.Y.shape[0, -1:]
    nu = h5_dataset.U.shape[-2:]
    train_state = create_train_state(network, ny, nu, options, init_rng)

    # Load the full dataset into GPU memory if we can. Otherwise we will load
    # superbatches one at a time in each epoch.
    if options.num_superbatches == 1:
        superbatch = h5_dataset[:]

    # Helper function for shuffling the dataset
    jit_shuffle = jax.jit(
        lambda data, perm: jax.tree_map(lambda x: x[perm], data)
    )

    metrics = {"loss": [], "load_time": [], "train_time": []}
    for epoch in range(options.epochs):
        rng, epoch_rng = jax.random.split(rng)
        perm = jax.random.permutation(epoch_rng, num_data_points)

        # Train on each superbatch
        load_time = 0.0
        train_time = 0.0
        for s in range(options.num_superbatches):
            st = time.time()
            if options.num_superbatches == 1:
                # The superbatch is already loaded, we'll just shuffle it
                superbatch = jit_shuffle(superbatch, perm)
            else:
                # Load the superbatch into GPU memory. This is the slowest part.
                idxs = perm[s * superbatch_size : (s + 1) * superbatch_size]
                superbatch = h5_dataset[idxs]
            jax.block_until_ready(superbatch)
            load_time += time.time() - st

            # Train on the superbatch
            st = time.time()
            train_state, loss = train_superbatch(
                train_state, superbatch, options
            )
            jax.block_until_ready(loss)
            train_time += time.time() - st

        metrics["loss"].append(loss)
        metrics["load_time"].append(load_time)
        metrics["train_time"].append(train_time)

        # Print some training statistics
        if (epoch + 1) % options.print_every == 0:
            print(
                f"Epoch {epoch + 1} / {options.epochs}, "
                f"loss: {loss:.4f}, "
                f"load time: {load_time:.4f}s, "
                f"train time: {train_time:.4f}s"
            )

    return train_state.params, metrics
