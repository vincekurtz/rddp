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
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
            The sampled control tapes Ũʲ, j = 1..M.
            The importance weights wₖ(Ũʲ).
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

        return score_estimate, U_noised, weights

    def generate_dataset(
        self, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate a dataset of noised score estimates from one initial state.

        The procedure for doing so is as follows:
          - Sample an initial state x₀
          - Sample a control tape μ = [u₀, u₁, ..., u_T₋₁] ~ 𝒩(0, σ_L²)
          - For each noise level k = L, L-1, ..., 0:
              - Sample Uₖⁱ ~ 𝒩(μₖ, σₖ²), i = 1..N
              - Estimate noised score ŝ = σₖ² ∇ log pₖ(Uₖⁱ | x₀) with M rollouts
              - Add (x₀, Uₖⁱ, ŝ, k) to the dataset
              - Update the mean control tape μₖ₋₁ = MPPI(Uₖⁱʲ)

        By the end of this process, μ₀ should be close to a local optimum.

        Args:
            rng: The random number generator key.

        Returns:
            Dataset of states, controls, scores, and noise levels (x₀, U, ŝ, k).
        """
        raise NotImplementedError
