import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import h5py
import jax
import jax.numpy as jnp
from brax.envs.base import State
from flax.struct import dataclass

from rddp.ocp import OptimalControlProblem
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

        # Save langevin sampling options, since we'll use them again when we
        # deploy the trained policy.
        save_path = Path(datagen_options.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "langevin_options.pkl", "wb") as f:
            pickle.dump(self.langevin_options, f)

        # Initialize the hdf5 file to save the dataset to
        self.h5_path = save_path / "dataset.h5"
        y_shape = (prob.env.observation_size,)
        U_shape = (prob.num_steps - 1, prob.env.action_size)
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
        x0: State,
        controls: jnp.ndarray,
        sigma: float,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
            The average cost of the rollouts
        """
        M = self.datagen_options.num_rollouts_per_data_point
        lmbda = self.datagen_options.starting_temperature * sigma**2

        # Sample control tapes Ũʲ ~ 𝒩(U,σ²)
        rng, ctrl_rng = jax.random.split(rng)
        U_noised = controls + sigma * jax.random.normal(
            ctrl_rng, (M, *controls.shape)
        )

        # Compute the cost of each control tape
        J, _ = jax.vmap(self.prob.rollout, in_axes=(None, 0))(x0, U_noised)
        avg_cost = jnp.mean(J, axis=0)
        J = J - jnp.min(J, axis=0)  # normalize for better numerics

        # Compute importance weights
        weights = jnp.exp(-J / lmbda)
        weights = weights / (jnp.sum(weights, axis=0) + 1e-6)  # avoid / 0

        # Compute the noised score estimate
        deltaU = U_noised - controls
        score_estimate = jnp.einsum("i,i...->...", weights, deltaU)
        score_estimate /= sigma**2

        return score_estimate, avg_cost

    def save_dataset(self, dataset: DiffusionDataset) -> None:
        """Add a dataset to the hdf5 file.

        Args:
            dataset: The dataset to save.
        """
        # Flatten the dataset for saving
        dataset = jax.tree.map(
            lambda x: jnp.reshape(x, (-1, *x.shape[2:])), dataset
        )

        # Write the dataset to the hdf5 file
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

    def initialize_dataset(
        self,
        y0: jnp.ndarray,
        controls: jnp.ndarray,
        score: jnp.ndarray,
        k: int,
        sigma: int,
    ) -> DiffusionDataset:
        """Initialize a diffusion dataset that we can add to.

        Adds an extra axis to the input arrays to make it easy to stack with
        new data later.

        Args:
            y0: The initial observation.
            controls: The control sequence.
            score: The noised score estimate.
            k: The noise level index.
            sigma: The noise level.

        Returns:
            The initialized dataset.
        """
        return DiffusionDataset(
            y0=y0[None],
            U=controls[None],
            s=score[None],
            k=k[None],
            sigma=sigma[None],
        )

    def add_to_dataset(
        self,
        dataset: DiffusionDataset,
        y0: jnp.ndarray,
        controls: jnp.ndarray,
        score: jnp.ndarray,
        k: int,
        sigma: int,
    ) -> DiffusionDataset:
        """Append new data to the dataset.

        Args:
            dataset: The existing dataset.
            y0: The initial observation.
            controls: The control sequence U.
            score: The noised score estimate.
            k: The noise level index.
            sigma: The noise level.

        Returns:
            The updated dataset.
        """
        return dataset.replace(
            y0=jnp.concatenate([dataset.y0, y0[None]]),
            U=jnp.concatenate([dataset.U, controls[None]]),
            s=jnp.concatenate([dataset.s, score[None]]),
            k=jnp.concatenate([dataset.k, k[None]]),
            sigma=jnp.concatenate([dataset.sigma, sigma[None]]),
        )

    def langevin_step(
        self,
        controls: jnp.ndarray,
        score: jnp.ndarray,
        sigma: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Perform a single Langevin step on the control tape.

            U_new = U + αs + β√(2α)ε,
            ε ~ N(0, I).

        Args:
            controls: The control tape U.
            score: The score estimate s.
            rng: The random number generator key.
        """
        alpha = self.langevin_options.step_size * sigma**2
        beta = self.langevin_options.noise_injection_level
        noise = jax.random.normal(rng, controls.shape)
        return controls + alpha * score + beta * jnp.sqrt(2 * alpha) * noise

    def generate(self, rng: jax.random.PRNGKey) -> DiffusionDataset:
        """Generate a dataset of noised score values and save it to disk.

        Args:
            rng: The random number generator key.
        """
        start_time = datetime.now()

        # Some useful shorthand parameters
        L = self.langevin_options.num_noise_levels
        N = self.datagen_options.num_initial_states
        sigmaL = self.langevin_options.starting_noise_level

        # Some helper functions
        # TODO: consider combining steps into one jitted function
        jit_reset = jax.jit(jax.vmap(self.prob.env.reset))
        jit_score = jax.jit(
            jax.vmap(self.estimate_noised_score, in_axes=(0, 0, None, None))
        )
        
        jit_initialize = jax.jit(
            lambda y0, U, s, k, sigma:
            self.initialize_dataset(y0, U, s, jnp.tile(k, (N, 1)), jnp.tile(sigma, (N, 1)))
        )
        jit_update = jax.jit(
            lambda dataset, y0, U, s, k, sigma:
            self.add_to_dataset(dataset, y0, U, s, jnp.tile(
                k, (N, 1)), jnp.tile(sigma, (N, 1)))
        )
            
        jit_langevin_step = jax.jit(
            jax.vmap(self.langevin_step, in_axes=(0, 0, None, None))
        )

        # Set the initial state
        rng, state_rng = jax.random.split(rng)
        state_rng = jax.random.split(state_rng, N)
        x0 = jit_reset(state_rng)

        # Sample inital control tape U ~ 𝒩(0, σ_L²)
        rng, init_rng = jax.random.split(rng)
        U = sigmaL * jax.random.normal(
            init_rng, (N, self.prob.num_steps - 1, self.prob.env.action_size)
        )

        # Initialize the dataset
        rng, score_rng = jax.random.split(rng)
        s, cost = jit_score(x0, U, sigmaL, score_rng)
        dataset = jit_initialize(x0.obs, U, s, L, sigmaL)

        # Print some debugging infos
        print(
            f"σₖ = {sigmaL:.4f}, "
            f"cost = {jnp.mean(cost):.4f} +/- {jnp.std(cost):.4f}, "
            f"time = {datetime.now() - start_time}"
        )

        for k in range(L - 1, -1, -1):
            rng, score_rng, step_rng = jax.random.split(rng, 3)

            # Set the noise level σₖ
            t = (L - k) / L
            sigma = self.langevin_options.starting_noise_level * jnp.exp(
                -self.langevin_options.noise_decay_rate * t
            )

            # Update the control tape from the previous score
            U = jit_langevin_step(U, s, sigma, step_rng)

            # Compute the score estimate for the new control tape
            s, cost = jit_score(x0, U, sigma, score_rng)

            # Update the dataset
            dataset = jit_update(dataset, x0.obs, U, s, k, sigma)

            print(
                f"σₖ = {sigma:.4f}, "
                f"cost = {jnp.mean(cost):.4f} +/- {jnp.std(cost):.4f}, "
                f"time = {datetime.now() - start_time}"
            )

            if k % self.datagen_options.save_every == 0:
                # Save what we have so far to avoid running out of memory
                print(f"**** Saving to {self.h5_path} at k={k} ****")
                self.save_dataset(dataset)
                dataset = jit_initialize(x0.obs, U, s, k, sigma)

        print(f"**** Saving to {self.h5_path} ****")
        self.save_dataset(dataset)