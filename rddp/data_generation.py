import pickle
from pathlib import PosixPath

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from rddp.tasks.base import OptimalControlProblem
from rddp.utils import (
    AnnealedLangevinOptions,
    DiffusionDataset,
    annealed_langevin_sample,
)


@dataclass
class DatasetGenerationOptions:
    """Parameters for generating a diffusion policy dataset.

    Attributes:
        save_path: The path to save the dataset to.
        temperature: The temperature λ of the target distribution.
        num_initial_states: The number of initial states x₀ to sample.
        noise_levels_per_file: The number of noise levels k to store per file.
        num_rollouts_per_data_point: The number of rollouts used to estimate
                                     each score, M.
    """

    save_path: PosixPath
    temperature: float
    num_initial_states: int
    noise_levels_per_file: int
    num_rollouts_per_data_point: int


class DatasetGenerator:
    """Generate a diffusion policy dataset for score function learning.

    The dataset consists of tuples
        (x₀, U, ŝ, k, σₖ),
    where
        x₀ is the initial state,
        U is the control sequence U = [u₀, u₁, ..., u_T₋₁],
        ŝ is the noised score estimate ŝ = ∇ log pₖ(U | x₀),
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

        # Ensure that we can split the dataset into equal-sized files
        assert langevin_options.num_noise_levels % datagen_options.noise_levels_per_file == 0
        self.num_files = langevin_options.num_noise_levels // datagen_options.noise_levels_per_file

        # Save langevin sampling options, since we'll use them again when we
        # deploy the trained policy.
        with open(datagen_options.save_path / "langevin_options.pkl", "wb") as f:
            pickle.dump(langevin_options, f)


    def estimate_noised_score(
        self,
        x0: jnp.ndarray,
        controls: jnp.ndarray,
        sigma: float,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Estimate the noised score s = ∇ log pₖ(U | x₀) with M rollouts.

        The score of the noised target distribution

            pₖ(U | x₀) = ∫ p(Ũ | x₀)N(Ũ;U,σₖ²)dŨ,
            p(U | x₀) ∝ exp(-J(U | x₀) / λ),

        is characterized by

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
        lmbda = self.datagen_options.temperature

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
        score_estimate /= sigma**2

        return score_estimate
    
    def generate_and_save(self, rng: jax.random.PRNGKey) -> None:
        """Generate and save a dataset of noised score estimates.

        The dataset is split into multiple files, each containing a subset of
        the noise levels, to avoid OOM errors when dealing with large systems
        or many initial conditions.

        Args:
            rng: The random number generator key.
        """
        sigmaL = self.langevin_options.starting_noise_level

        # Some helper functions
        sample_initial_state = jax.jit(jax.vmap(self.prob.sample_initial_state))
        langevin_sample = jax.vmap(
            lambda x0, U, rng, noise_range:
            annealed_langevin_sample(
                self.langevin_options,
                x0,
                U,
                self.estimate_noised_score,
                rng,
                noise_range,
            ),
            in_axes=(0, 0, 0, None),
        )
        calc_cost = jax.jit(jax.vmap(self.prob.total_cost))
        
        # Sample initial states
        rng, state_rng = jax.random.split(rng)
        state_rng = jax.random.split(
            state_rng, self.datagen_options.num_initial_states
        )
        x0 = sample_initial_state(state_rng)

        # Sample inital control tapes U ~ 𝒩(0, σ_L²)
        rng, init_rng = jax.random.split(rng)
        U = sigmaL * jax.random.normal(
            init_rng, (self.datagen_options.num_initial_states,
                       self.prob.num_steps - 1, *self.prob.sys.action_shape)
        )

        costs = calc_cost(U, x0)
        print(f"σₖ = {sigmaL:.4f}, cost = {jnp.mean(costs):.4f} +/- {jnp.std(costs):.4f}")

        for i in range(self.num_files, 0, -1):
            start_k = i * self.datagen_options.noise_levels_per_file
            end_k = (i - 1) * self.datagen_options.noise_levels_per_file

            # Generate data with annealed Langevin sampling at the given noise
            # levels.
            rng, langevin_rng = jax.random.split(rng)
            langevin_rng = jax.random.split(
                langevin_rng, self.datagen_options.num_initial_states
            )
            U, dataset = langevin_sample(x0, U, langevin_rng, (start_k, end_k))

            # Print a quick performance summary
            costs = calc_cost(U, x0)
            sigma = dataset.sigma[0, -1, 0, 0]  # state, noise level, step, dim
            print(f"σₖ = {sigma:.4f}, cost = {jnp.mean(costs):.4f} +/- {jnp.std(costs):.4f}")

            # Save the dataset to a file
            with open(self.datagen_options.save_path / f"langevin_data_{i}.pkl", "wb") as f:
                pickle.dump(dataset, f)


    def generate_from_state(
        self, x0: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> DiffusionDataset:
        """Generate a dataset of noised score estimates from one initial state.

        Args:
            x0: The initial state x₀.
            rng: The random number generator key.

        Returns:
            Dataset of states, controls, scores, and noise levels (x₀, U, ŝ, k).
        """
        sigmaL = self.langevin_options.starting_noise_level

        # Sample U ~ 𝒩(0, σ_L²)
        rng, mu_rng = jax.random.split(rng)
        U = sigmaL * jax.random.normal(
            mu_rng, (self.prob.num_steps - 1, *self.prob.sys.action_shape)
        )

        # Generate data for each noise level
        _, dataset = annealed_langevin_sample(
            options=self.langevin_options,
            x0=x0,
            u_init=U,
            score_fn=self.estimate_noised_score,
            rng=rng,
        )

        return dataset

    def generate(self, rng: jax.random.PRNGKey) -> DiffusionDataset:
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
        stacked_data = jax.vmap(self.generate_from_state)(x0s, gen_rng)

        # Flatten the data across initial states, noise levels, and data points.
        flat_data = jax.tree.map(
            lambda x: jnp.reshape(x, (-1, *x.shape[3:])), stacked_data
        )

        return flat_data

    def save_dataset(self, dataset: DiffusionDataset, path: str) -> None:
        """Save a training dataset to a file.

        Note that this also saves the annealed langevin options, which are
        needed for depolying a trained score model.

        Args:
            dataset: The dataset to save.
            path: The path to save the dataset to.
        """
        data = {"dataset": dataset, "langevin_options": self.langevin_options}
        with open(path, "wb") as f:
            pickle.dump(data, f)

