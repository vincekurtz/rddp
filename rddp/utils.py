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
        x0: The initial state x₀.
        U: The control sequence U = [u₀, u₁, ..., u_T₋₁].
        s: The noised score estimate ŝ = ∇ log pₖ(U | x₀).
        k: The noise level index k.
        sigma: The noise level σₖ.
    """

    x0: jnp.ndarray
    U: jnp.ndarray
    s: jnp.ndarray
    k: jnp.ndarray
    sigma: jnp.ndarray


@dataclass
class AnnealedLangevinOptions:
    """Parameters for annealed Langevin dynamics.

    Annealed Langevin dynamics samples from the target distribution

        p(U | x₀) ∝ exp(-J(U | x₀) / λ),

    by considering intermediate noised distributions

        pₖ(U | x₀) = ∫ p(Ũ | x₀)N(Ũ;U,σₖ²)dŨ

    with a decreasing sequence of noise levels σₖ.

    Attributes:
        num_noise_levels: The number of noise levels L.
        starting_noise_level: The starting noise level σ_L.
        num_steps: The number of Langevin steps to take at each noise level, N.
        step_size: The Langevin step size α.
        noise_injection_level: The noise injection level for each Langevin step.
            A value of 1.0 corresponds to the standard Langevin dynamics, while
            a value of 0.0 corresponds to pure gradient descent.
    """

    num_noise_levels: int
    starting_noise_level: int
    num_steps: int
    step_size: float
    noise_injection_level: float = 1.0


def annealed_langevin_sample(
    options: AnnealedLangevinOptions,
    x0: jnp.ndarray,
    u_init: jnp.ndarray,
    score_fn: Callable[
        [jnp.ndarray, jnp.ndarray, float, jax.random.PRNGKey], jnp.ndarray
    ],
    rng: jax.random.PRNGKey,
    noise_range: Tuple[int, int] = None,
) -> Tuple[jnp.ndarray, DiffusionDataset]:
    """Generate a sample from the target distribution p(U | x₀).

    Annealed Langevin samples intermediate distributions

        pₖ(U | x₀) = ∫ p(Ũ | x₀)N(Ũ;U,σₖ²)dŨ

    with a decreasing sequence of noise levels σₖ. At each level, we sample
    from pₖ(U | x₀) using Langevin dynamics:

        Uⁱ⁺¹ = Uⁱ + ε ŝ(x₀, Uⁱ, σₖ) + √(2ε) zⁱ,

    where ŝ(x₀, Uⁱ, σₖ) is an estimate of the score ∇ log pₖ(U | x₀),
    ε = ασₖ² is the step size, and zⁱ ~ N(0, I) is Gaussian noise.

    Args:
        options: The annealed Langevin options defining α, σₖ, etc.
        x0: The initial state x₀ that the target distribution is conditioned on.
        u_init: An initial control sequence, typically U ~ N(0, σ_L²).
        score_fn: A (possibly stochastic) score estimate function ŝ(x₀, U, σ).
        rng: The random number generator key.
        noise_range: The range of noise levels to sample from. If None, sample
            from L to 0. This option is useful for dataset generation, where we
            want to save out to a file during the annealing process.
    """
    L = options.num_noise_levels
    sigmaL = options.starting_noise_level

    if noise_range is None:
        start_step, end_step = L, 0
    else:
        start_step, end_step = noise_range
        assert start_step >= end_step, "start_step should be >= end_step"
        assert start_step <= L, "start_step should be <= L"
        assert end_step >= 0, "end_step should be >= 0"

    N = options.num_steps
    alpha = options.step_size
    beta = options.noise_injection_level

    def langevin_step(carry: Tuple, i: int):
        """Perform a single Langevin sampling step at the k-th noise level.

        Return the new control tape Uₖⁱ⁺¹ and the score estimate ŝₖⁱ.
        """
        U, sigma, k, rng = carry
        rng, score_rng, z_rng = jax.random.split(rng, 3)
        eps = alpha * sigma**2

        # Langevin dynamics based on the estimated score
        z = jax.random.normal(z_rng, U.shape)
        s = score_fn(x0, U, sigma, score_rng)
        U_new = U + eps * s + beta * jnp.sqrt(2 * eps) * z

        # Record training data
        data = DiffusionDataset(
            x0=x0,
            U=U,
            s=s,
            k=jnp.array([k]),
            sigma=jnp.array([sigma]),
        )

        return (U_new, sigma, k, rng), data

    def annealed_langevin_step(carry: Tuple, k: int):
        """Generate N samples at the k-th noise level."""
        U, rng = carry

        # Set the noise level σₖ
        t = (L - k) / L
        sigma = sigmaL * jnp.exp(-4 * t)

        # Run Langevin dynamics for N steps, recording score estimates
        # along the way
        rng, langevin_rng = jax.random.split(rng)
        (U, _, _, _), data = jax.lax.scan(
            langevin_step, (U, sigma, k, langevin_rng), jnp.arange(N)
        )

        return (U, rng), data

    rng, sampling_rng = jax.random.split(rng)
    (U, _), dataset = jax.lax.scan(
        annealed_langevin_step,
        (u_init, sampling_rng),
        jnp.arange(start_step - 1, end_step - 1, -1),
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
        dataset.x0.shape[0],
        1,
    ), "dataset should be flattened"

    idxs = jnp.where(dataset.k == k)[0]
    rng, sample_rng = jax.random.split(rng)
    idxs = jax.random.permutation(sample_rng, idxs)
    idxs = idxs[:num_samples]

    return DiffusionDataset(
        x0=dataset.x0[idxs],
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

        Args:
            file: The HDF5 file. Must remain open for the lifetime of
                  this object.
        """
        self.x0 = np.array(file["x0"])
        self.U = np.array(file["U"])
        self.s = np.array(file["s"])
        self.sigma = np.array(file["sigma"])
        self.k = np.array(file["k"])

        # Size checks
        self.num_data_points = self.x0.shape[0]
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

        However, note that h5py slicing requires that indices are sorted.
        See https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing

        Args:
            idx: The index of the data point to extract.

        Returns:
            A jax dataset object containing the data at the given indices.
        """
        return DiffusionDataset(
            x0=jnp.array(self.x0[idx]),
            U=jnp.array(self.U[idx]),
            s=jnp.array(self.s[idx]),
            k=jnp.array(self.k[idx]),
            sigma=jnp.array(self.sigma[idx]),
        )
