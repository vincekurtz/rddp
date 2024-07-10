from pathlib import Path
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState

from rddp.data_loading import TorchDiffusionDataLoader, TorchDiffusionDataset

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
def apply_model(state: TrainState, batch: dict) -> Tuple[jnp.ndarray, Params]:
    """Compute the loss and gradients for a batch of data points.

    Args:
        state: The training state, including the network and parameters.
        batch: The batch of training data (x0, U, sigma, s).

    Returns:
        The training loss and parameter gradients.
    """

    def loss_fn(params: Params) -> jnp.ndarray:
        """Compute the loss for a batch of data points."""
        s = state.apply_fn(params, batch["x0"], batch["U"], batch["sigma"])
        err = jnp.square(s - batch["s"])

        # Weigh the error by the noise level, as recommended by Song et al.
        # Generative Modeling by Estimating Gradients of the Data Distribution,
        # NeurIPS 2019.
        err = jnp.einsum("ij,i...->i...", batch["sigma"] ** 2, err)

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


def train(
    network: nn.Module,  # TODO: make a custom score network type
    data_path: Union[str, Path],
    options: TrainingOptions,
    seed: int = 0,
) -> Tuple[Params, dict]:
    """Train a score network on the given dataset.

    Args:
        network: The score network to train.
        data_path: The path to the saved training dataset
        options: The training options.
        seed: The random seed.

    Returns:
        The trained score network parameters and some training metrics.
    """
    rng = jax.random.PRNGKey(seed)
    data_path = Path(data_path)

    # Create a torch dataset and data loader
    dataset = TorchDiffusionDataset(data_path)
    data_loader = TorchDiffusionDataLoader(
        dataset, batch_size=options.batch_size, shuffle=True
    )

    # Initialize the training state
    rng, init_rng = jax.random.split(rng)
    nx = dataset[0]["x0"].shape[-1:]  # TODO: support more generic shapes
    nu = dataset[0]["U"].shape[-2:]
    train_state = create_train_state(network, nx, nu, options, init_rng)

    # Train the model
    metrics = {"train_loss": [], "val_loss": []}
    for epoch in range(options.epochs):
        for batch in data_loader:
            loss, grad = apply_model(train_state, batch)
            train_state = update_model(train_state, grad)

        # TODO: compute some sort of validation loss

        print(f"Epoch {epoch}, loss {loss:.4f}")
        metrics["train_loss"].append(loss)

    return train_state.params, metrics
