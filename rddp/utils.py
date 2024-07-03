from typing import Callable, Tuple

import jax
import jax.numpy as jnp
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

    with a geometrically decreasing sequence of noise levels k = L, L-1, ..., 0.

    Attributes:
        num_noise_levels: The number of noise levels L.
        starting_noise_level: The starting noise level σ_L.
        noise_decay_rate: The noise decay rate σₖ₋₁ = γ σₖ.
        num_steps: The number of Langevin steps to take at each noise level, N.
        step_size: The Langevin step size α.
        noise_injection_level: The noise injection level for each Langevin step.
            A value of 1.0 corresponds to the standard Langevin dynamics, while
            a value of 0.0 corresponds to pure gradient descent.
    """

    num_noise_levels: int
    starting_noise_level: int
    noise_decay_rate: float
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
    """
    L = options.num_noise_levels
    sigmaL = options.starting_noise_level
    gamma = options.noise_decay_rate
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
        (U, sigma, rng) = carry

        # Run Langevin dynamics for N steps, recording score estimates
        # along the way
        rng, langevin_rng = jax.random.split(rng)
        (U, _, _, _), data = jax.lax.scan(
            langevin_step, (U, sigma, k, langevin_rng), jnp.arange(N)
        )

        # Reduce the noise level σₖ₋₁ = γ σₖ
        sigma *= gamma

        return (U, sigma, rng), data

    rng, sampling_rng = jax.random.split(rng)
    (U, _, _), dataset = jax.lax.scan(
        annealed_langevin_step,
        (u_init, sigmaL, sampling_rng),
        jnp.arange(L - 1, -1, -1),
    )

    return U, dataset
