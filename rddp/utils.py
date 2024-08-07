from typing import Callable, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass


@dataclass
class DiffusionDataset:
    """Training data for a diffusion policy.

    Attributes:
        y0: The initial observation y₀.
        U: The control sequence U = [u₀, u₁, ..., u_T₋₁].
        s: The noised score estimate ŝ = ∇ log pₖ(U | y₀).
        k: The noise level index k.
        sigma: The noise level σₖ.
    """

    y0: jnp.ndarray
    U: jnp.ndarray
    s: jnp.ndarray
    k: jnp.ndarray
    sigma: jnp.ndarray


@dataclass
class AnnealedLangevinOptions:
    """Parameters for annealed Langevin dynamics.

    Annealed Langevin dynamics samples from the target distribution

        p(U | y₀) ∝ exp(-J(U | y₀) / λ),

    by considering intermediate noised distributions

        pₖ(U | y₀) = ∫ p(Ũ | y₀)N(Ũ;U,σₖ²)dŨ

    with a decreasing sequence of noise levels σₖ.

    Attributes:
        num_noise_levels: The number of noise levels L.
        starting_noise_level: The starting noise level σ_L.
        step_size: The Langevin step size α.
        noise_injection_level: The noise injection level for each Langevin step.
            A value of 1.0 corresponds to the standard Langevin dynamics, while
            a value of 0.0 corresponds to pure gradient descent.
        noise_decay_rate: The rate β at which the noise level decays, σₖ = σ_L
            exp(-βt), where t = (L - k) / L.
    """

    num_noise_levels: int
    starting_noise_level: int
    step_size: float
    noise_injection_level: float = 1.0
    noise_decay_rate: float = 4.0


def annealed_langevin_sample(
    options: AnnealedLangevinOptions,
    y0: jnp.ndarray,
    controls: jnp.ndarray,
    score_fn: Callable[
        [jnp.ndarray, jnp.ndarray, float, jax.random.PRNGKey], jnp.ndarray
    ],
    rng: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, DiffusionDataset]:
    """Generate a sample from the target distribution p(U | y₀).

    Annealed Langevin samples intermediate distributions

        pₖ(U | y₀) = ∫ p(Ũ | y₀)N(Ũ;U,σₖ²)dŨ

    with a decreasing sequence of noise levels σₖ. At each step, we update the
    control sequences Uᵏ as

        Uᵏ⁺¹ = Uᵏ + αŝ(y₀, Uⁱ, σₖ) + β√(2α)ε,
        ε ~ N(0, I).

    where ŝ(y₀, Uⁱ, σₖ) is an estimate of the score ∇ log pₖ(U | y₀),
    α ∝ σₖ² is the step size, and β is the noise injection level.

    Args:
        options: The annealed Langevin options defining α, σₖ, etc.
        y0: The initial observation y₀ that we condition on.
        controls: An initial control sequence, typically U ~ N(0, σ_L²).
        score_fn: A (possibly stochastic) score estimate function ŝ(y₀, U, σ).
        rng: The random number generator key.

    Returns:
        The final sample U⁰ ~ p(U | y₀).
        A dataset containing the intermediate control sequences Uᵏ scores, etc.
    """
    L = options.num_noise_levels
    sigmaL = options.starting_noise_level
    beta = options.noise_injection_level

    def step_fn(carry: Tuple, k: int):
        """Update Uᵏ⁺¹ = Uᵏ + αŝ(y₀, Uⁱ, σₖ) + β√(2α)ε."""
        U, rng = carry
        rng, score_rng, noise_rng = jax.random.split(rng, 3)

        # Set the noise level σₖ and step size α
        t = (L - k) / L
        sigma = sigmaL * jnp.exp(-options.noise_decay_rate * t)
        alpha = options.step_size * sigma**2

        # Estimate the score ∇ log pₖ(U | y₀)
        s = score_fn(y0, U, sigma, score_rng)

        # Take a Langevin step
        noise = jax.random.normal(noise_rng, U.shape)
        U_new = U + alpha * s + beta * jnp.sqrt(2 * alpha) * noise

        # Save intermediate values in a diffusion dataset
        dataset = DiffusionDataset(
            y0=y0,
            U=U,
            s=s,
            k=jnp.full((U.shape[0], 1), k),
            sigma=jnp.full((U.shape[0], 1), sigma),
        )

        return (U_new, rng), dataset

    (U, _), dataset = jax.lax.scan(
        step_fn,
        (controls, rng),
        jnp.arange(L - 1, -1, -1),
    )

    return U, dataset


def sample_dataset(
    dataset: DiffusionDataset, k: int, num_samples: int, rng: jax.random.PRNGKey
) -> DiffusionDataset:
    """Extract some random samples from the dataset at a specific noise level.

    This is particuarly useful for visualizing the training dataset.

    Args:
        dataset: The full dataset.
        k: The noise level index.
        num_samples: The number of samples to extract.
        rng: The random number generator key to use for sampling.

    Returns:
        A subset of the dataset at the given noise level.
    """
    assert dataset.k.shape == (
        dataset.y0.shape[0],
        1,
    ), "dataset should be flattened"

    idxs = jnp.where(dataset.k == k)[0]
    rng, sample_rng = jax.random.split(rng)
    idxs = jax.random.permutation(sample_rng, idxs)
    idxs = idxs[:num_samples]

    return DiffusionDataset(
        y0=dataset.y0[idxs],
        U=dataset.U[idxs],
        s=dataset.s[idxs],
        k=dataset.k[idxs],
        sigma=dataset.sigma[idxs],
    )


class HDF5DiffusionDataset:
    """A wrapper around an HDF5 file containing a diffusion dataset.

    Provides a simple interface reading data that is stored on disc in an HDF5
    file. This is essential for working with large datasets that do not fit
    into memory.
    """

    def __init__(self, file: h5py.File):
        """Initialize the dataset wrapper.

        Note that this loads the entire dataset into CPU memory.

        Args:
            file: The HDF5 file. Must be open in read mode on construction.
        """
        # Load the data from the HDF5 file into CPU memory. For some reason
        # conversion to jnp arrays is super slow when done directly from the
        # HDF5 file, so we load everything into CPU memory first and only move
        # to GPU when the data is accessed with __getitem__.
        self.y0 = np.array(file["y0"], dtype=jnp.float32)
        self.U = np.array(file["U"], dtype=jnp.float32)
        self.s = np.array(file["s"], dtype=jnp.float32)
        self.sigma = np.array(file["sigma"], dtype=jnp.float32)
        self.k = np.array(file["k"], dtype=jnp.int32)

        # Size checks
        self.num_data_points = self.y0.shape[0]
        assert self.U.shape[0] == self.num_data_points
        assert self.s.shape[0] == self.num_data_points
        assert self.sigma.shape[0] == self.num_data_points
        assert self.k.shape[0] == self.num_data_points
        assert self.U.shape == self.s.shape
        assert self.sigma.shape == self.k.shape

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return self.num_data_points

    def __getitem__(self, idx: int) -> DiffusionDataset:
        """Load the data at the given indices into GPU memory.

        This allows us to access the data with slicing syntax, e.g.,

            my_jax_batch = my_hdf5_dataset[10:20]

        Args:
            idx: The index of the data point to extract.

        Returns:
            A jax dataset object containing the data at the given indices.
        """
        return DiffusionDataset(
            y0=jnp.asarray(self.y0[idx], dtype=jnp.float32),
            U=jnp.asarray(self.U[idx], dtype=jnp.float32),
            s=jnp.asarray(self.s[idx], dtype=jnp.float32),
            sigma=jnp.asarray(self.sigma[idx], dtype=jnp.float32),
            k=jnp.asarray(self.k[idx], dtype=jnp.int32),
        )
