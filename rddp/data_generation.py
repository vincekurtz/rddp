from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from rddp.tasks.base import OptimalControlProblem


@dataclass
class AnnealedLangevinOptions:
    """Parameters for annealed Langevin dynamics.

    Annealed Langevin dynamics samples from the target distribution

        p(U | x₀) ∝ exp(-J(U | x₀) / λ),

    by considering intermediate noised distributions

        pₖ(U | x₀) = ∫ p(Ũ | x₀)N(Ũ;U,σₖ²)dŨ

    with a geometrically decreasing sequence of noise levels k = L, L-1, ..., 0.

    Attributes:
        temperature: The temperature λ
        num_noise_levels: The number of noise levels L.
        starting_noise_level: The starting noise level σ_L.
        noise_decay_rate: The noise decay rate σₖ₋₁ = γ σₖ.
    """

    temperature: float
    num_noise_levels: int
    starting_noise_level: int
    noise_decay_rate: float


@dataclass
class DatasetGenerationOptions:
    """Parameters for generating a diffusion policy dataset.

    Attributes:
        num_initial_states: The number of initial states x₀ to sample.
        num_data_points_per_initial_state: The number of data points per initial
                                           state, N.
        num_rollouts_per_data_point: The number of rollouts used to estimate
                                     each score, M.
    """

    num_initial_states: int
    num_data_points_per_initial_state: int
    num_rollouts_per_data_point: int


class DatasetGenerator:
    """Generate a diffusion policy dataset for score function learning.

    The dataset consists of tuples
        (x₀, U, ŝ, k),
    where
        x₀ is the initial state,
        U is the control sequence U = [u₀, u₁, ..., u_T₋₁],
        ŝ is the noised score estimate ŝ = σₖ² ∇ log pₖ(U | x₀),
        and k is noise level index.
    """

    def __init__(
        self,
        prob: OptimalControlProblem,
        langevin_options: AnnealedLangevinOptions,
        datagen_options: DatasetGenerationOptions,
    ):
        """Initialize the dataset generator.

        Args:
            prob: The optimal control problem defining the cost J(U | x₀).
            langevin_options: Sampling (e.g., temperature) settings.
            datagen_options: Dataset generation (e.g., num rollouts) settings.
        """
        self.prob = prob
        self.langevin_options = langevin_options
        self.datagen_options = datagen_options

    def estimate_noised_score(
        self,
        x0: jnp.ndarray,
        controls: jnp.ndarray,
        sigma: float,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Estimate the noised score s = σ² ∇ log pₖ(U | x₀) with M rollouts.

        The score of the noised target distribution

            pₖ(U | x₀) = ∫ p(Ũ | x₀)N(Ũ;U,σₖ²)dŨ,
            p(U | x₀) ∝ exp(-J(U | x₀) / λ),

        is given by

            σ² ∇ log pₖ(U | x₀) =
                𝔼[exp(-J(Ũ | x₀) / λ)(Ũ - U)] / 𝔼[exp(-J(Ũ | x₀) / λ)],

        where the expectation is under Ũ ~ 𝒩(U,σₖ²).

        Args:
            x0: The initial state x₀.
            controls: The control sequence U = [u₀, u₁, ..., u_T₋₁].
            sigma: The noise level σₖ.
            rng: The random number generator key.

        Returns:
            The noised score estimate ŝ = σ² ∇ log pₖ(U | x₀).
        """
        M = self.datagen_options.num_rollouts_per_data_point
        lmbda = self.langevin_options.temperature

        # Sample control tapes Ũʲ ~ 𝒩(U,σ²)
        rng, ctrl_rng = jax.random.split(rng)
        U_noised = controls + sigma * jax.random.normal(
            ctrl_rng, (M, *controls.shape)
        )

        # Compute the cost of each control tape
        J = jax.vmap(self.prob.total_cost, in_axes=(0, None))(U_noised, x0)
        J = J - jnp.min(J, axis=0)  # normalize for better numerics

        # Compute importance weights
        weights = jnp.exp(-J / lmbda)
        weights = weights / (jnp.sum(weights, axis=0) + 1e-6)  # avoid / 0

        # Compute the noised score estimate
        deltaU = U_noised - controls
        score_estimate = jnp.einsum("i,i...->...", weights, deltaU)

        return score_estimate

    def generate_from_state(
        self, x0: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate a dataset of noised score estimates from one initial state.

        Starting from initial state x₀:
          - Sample a control tape μ_L = [u₀, u₁, ..., u_T₋₁] ~ 𝒩(0, σ_L²)
          - For each noise level k = L, L-1, ..., 0:
              - Sample Uₖⁱ ~ 𝒩(μₖ, σₖ²), i = 1..N
              - Estimate noised score ŝ = σₖ² ∇ log pₖ(Uₖⁱ | x₀) with M rollouts
              - Add (x₀, Uₖⁱ, ŝₖⁱ, k) to the dataset
              - Update the mean control tape μₖ₋₁ = μₖ₋₁ + 1/N ∑ᵢ ŝₖⁱ

        By the end of this process, μ₀ should be close to a local optimum.

        Args:
            x0: The initial state x₀.
            rng: The random number generator key.

        Returns:
            Dataset of states, controls, scores, and noise levels (x₀, U, ŝ, k).
        """
        L = self.langevin_options.num_noise_levels
        N = self.datagen_options.num_data_points_per_initial_state
        sigmaL = self.langevin_options.starting_noise_level
        gamma = self.langevin_options.noise_decay_rate

        # Sample μ_L ~ 𝒩(0, σ_L²)
        rng, mu_rng = jax.random.split(rng)
        muL = sigmaL * jax.random.normal(
            mu_rng, (self.prob.num_steps - 1, *self.prob.sys.action_shape)
        )

        def scan_fn(carry: Tuple, k: int):
            """Estimate scores at the k-th noise level."""
            (mu, sigma, rng) = carry

            # Sample N control tapes Uₖⁱ ~ 𝒩(μₖ, σₖ²)
            rng, ctrl_rng = jax.random.split(rng)
            U = mu + sigma * jax.random.normal(ctrl_rng, (N, *mu.shape))

            # Estimate noised scores ŝ = σₖ² ∇ log pₖ(U | x₀)
            rng, score_rng = jax.random.split(rng)
            score_rng = jax.random.split(score_rng, N)
            s = jax.vmap(
                self.estimate_noised_score, in_axes=(None, 0, None, 0)
            )(x0, U, sigma, score_rng)

            # Update μₖ₋₁ by descending the score gradient
            # TODO: figure out a better/more principled update
            mu = mu + jnp.mean(s, axis=0)

            # Update σₖ₋₁ = γ σₖ
            sigma *= gamma

            # Put together the dataset (x₀, Uₖⁱ, ŝₖⁱ, k)
            dataset = (jnp.tile(x0, (N, 1)), U, s, jnp.full((N,), k))

            return (mu, sigma, rng), dataset

        rng, rollouts_rng = jax.random.split(rng)
        _, dataset = jax.lax.scan(
            scan_fn, (muL, sigmaL, rollouts_rng), jnp.arange(L - 1, -1, -1)
        )

        return dataset

    def generate(
        self, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate a dataset of noised score estimates, (x₀, U, ŝ, k).

        Data is generated for various initial conditions and noise levels, but
        flattened into a single dataset with shape [sample, data].

        Args:
            x0: The initial state x₀.
            rng: The random number generator key.

        Returns:
            Dataset of states, controls, scores, and noise levels (x₀, U, ŝ, k).
        """
        # Sample initial states
        rng, state_rng = jax.random.split(rng)
        state_rng = jax.random.split(
            state_rng, self.datagen_options.num_initial_states
        )
        x0s = jax.vmap(self.prob.sample_initial_state)(state_rng)

        # Generate data for each initial state
        rng, gen_rng = jax.random.split(rng)
        gen_rng = jax.random.split(
            gen_rng, self.datagen_options.num_initial_states
        )
        x0, U, s, k = jax.vmap(self.generate_from_state)(x0s, gen_rng)

        # Flatten the data across initial states, noise levels, and data points.
        x0 = jnp.reshape(x0, (-1, *x0.shape[3:]))
        U = jnp.reshape(U, (-1, *U.shape[3:]))
        s = jnp.reshape(s, (-1, *s.shape[3:]))
        k = jnp.reshape(k, (-1, *k.shape[3:]))

        return x0, U, s, k
