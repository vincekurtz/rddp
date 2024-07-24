import pickle
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
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
        starting_temperature: The initial temperature λ.
        num_initial_states: The number of initial states x₀ to sample.
        num_rollouts_per_data_point: The number of rollouts used to estimate
                                     each score, M.
        save_every: The number of noise levels to generate between saves.
        save_path: The directory to save the generated dataset to.
    """

    starting_temperature: float
    num_initial_states: int
    num_rollouts_per_data_point: int
    save_every: int
    save_path: Union[str, Path]


class DatasetGenerator:
    """Generate a diffusion policy dataset for score function learning.

    The dataset consists of tuples
        (y₀, U, ŝ, k, σₖ),
    where
        y₀ is the initial observation,
        U is the control sequence U = [u₀, u₁, ..., u_T₋₁],
        ŝ is the noised score estimate ŝ = ∇ log pₖ(U | y₀),
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
            prob: The optimal control problem defining the cost J(U | y₀).
            langevin_options: Sampling (e.g., temperature) settings.
            datagen_options: Dataset generation (e.g., num rollouts) settings.
        """
        self.prob = prob
        self.langevin_options = langevin_options
        self.datagen_options = datagen_options

        # Ensure that we can split the dataset into equal-sized files
        assert (
            langevin_options.num_noise_levels % datagen_options.save_every == 0
        )
        self.num_saves = (
            langevin_options.num_noise_levels // datagen_options.save_every
        )

        # Save langevin sampling options, since we'll use them again when we
        # deploy the trained policy.
        save_path = Path(datagen_options.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "langevin_options.pkl", "wb") as f:
            pickle.dump(self.langevin_options, f)

        # Initialize the hdf5 file to save the dataset to
        self.h5_path = save_path / "dataset.h5"
        y_shape = prob.sys.observation_shape
        U_shape = (prob.num_steps - 1, *prob.sys.action_shape)
        with h5py.File(self.h5_path, "w") as f:
            f.create_dataset(
                "y0", (0, *y_shape), maxshape=(None, *y_shape), dtype="float32"
            )
            f.create_dataset(
                "U", (0, *U_shape), maxshape=(None, *U_shape), dtype="float32"
            )
            f.create_dataset(
                "s", (0, *U_shape), maxshape=(None, *U_shape), dtype="float32"
            )
            f.create_dataset("k", (0, 1), maxshape=(None, 1), dtype="int32")
            f.create_dataset(
                "sigma", (0, 1), maxshape=(None, 1), dtype="float32"
            )

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

        Note that we anneal the temperature λ along with the noise level σₖ.

        Args:
            x0: The initial state x₀.
            controls: The control sequence U = [u₀, u₁, ..., u_T₋₁].
            sigma: The noise level σₖ.
            rng: The random number generator key.

        Returns:
            The noised score estimate ŝ = σ² ∇ log pₖ(U | x₀).
        """
        M = self.datagen_options.num_rollouts_per_data_point
        lmbda = self.datagen_options.starting_temperature * sigma**2

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

    def save_dataset(self, dataset: DiffusionDataset) -> None:
        """Add a dataset to the hdf5 file.

        Args:
            dataset: The (flattened) dataset to save.
        """
        with h5py.File(self.h5_path, "a") as f:
            y0, U, s, k, sigma = f["y0"], f["U"], f["s"], f["k"], f["sigma"]
            num_existing_data_points = y0.shape[0]
            num_new_data_points = dataset.y0.shape[0]
            new_size = num_existing_data_points + num_new_data_points

            # Resize datasets to accomodate new data
            y0.resize(new_size, axis=0)
            U.resize(new_size, axis=0)
            s.resize(new_size, axis=0)
            k.resize(new_size, axis=0)
            sigma.resize(new_size, axis=0)

            # Write the new data
            y0[num_existing_data_points:] = dataset.y0
            U[num_existing_data_points:] = dataset.U
            s[num_existing_data_points:] = dataset.s
            k[num_existing_data_points:] = dataset.k
            sigma[num_existing_data_points:] = dataset.sigma

    def generate_and_save(self, rng: jax.random.PRNGKey) -> None:
        """Generate a dataset of noised score values and save it to disk.

        Args:
            rng: The random number generator key.
        """
        start_time = datetime.now()

        # Some helper functions
        sample_initial_state = jax.jit(jax.vmap(self.prob.sample_initial_state))
        langevin_sample = jax.vmap(
            lambda x0, u, rng, noise_range: annealed_langevin_sample(
                self.langevin_options,
                x0,  # N.B. we assume that p(U | x₀) = p(U | y₀)
                u,
                self.estimate_noised_score,
                rng,
                noise_range,
            ),
            in_axes=(0, 0, 0, None),
        )
        calc_cost = jax.jit(jax.vmap(self.prob.total_cost))
        calc_obs = jax.jit(jax.vmap(self.prob.sys.g))
        sigmaL = self.langevin_options.starting_noise_level

        # Sample initial states
        rng, state_rng = jax.random.split(rng)
        state_rng = jax.random.split(
            state_rng, self.datagen_options.num_initial_states
        )
        x0 = sample_initial_state(state_rng)

        # Sample inital control tapes U ~ 𝒩(0, σ_L²)
        rng, init_rng = jax.random.split(rng)
        U = sigmaL * jax.random.normal(
            init_rng,
            (
                self.datagen_options.num_initial_states,
                self.prob.num_steps - 1,
                *self.prob.sys.action_shape,
            ),
        )

        costs = calc_cost(U, x0)
        print(
            f"σₖ = {sigmaL:.4f}, "
            f"cost = {jnp.mean(costs):.4f} +/- {jnp.std(costs):.4f}, "
            f"time = {datetime.now() - start_time}"
        )

        for i in range(self.num_saves, 0, -1):
            start_k = i * self.datagen_options.save_every
            end_k = (i - 1) * self.datagen_options.save_every

            # Generate data with annealed Langevin sampling at the given noise
            # levels.
            rng, langevin_rng = jax.random.split(rng)
            langevin_rng = jax.random.split(
                langevin_rng, self.datagen_options.num_initial_states
            )
            U, dataset = langevin_sample(x0, U, langevin_rng, (start_k, end_k))

            # Flatten the dataset, then transform observations to y = g(x)
            flat_data = jax.tree.map(
                lambda x: jnp.reshape(x, (-1, *x.shape[3:])), dataset
            )
            flat_data = flat_data.replace(y0=calc_obs(flat_data.y0))
            self.save_dataset(flat_data)

            # Print a quick performance summary
            costs = calc_cost(U, x0)
            sigma = dataset.sigma[0, -1, 0, 0]  # state, noise level, step, dim
            print(
                f"σₖ = {sigma:.4f}, "
                f"cost = {jnp.mean(costs):.4f} +/- {jnp.std(costs):.4f}, "
                f"time = {datetime.now() - start_time}"
            )
