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
from rddp.utils import AnnealedLangevinOptions, DiffusionDataset


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
        print_every: How often to print a performance summary during generation.
    """

    starting_temperature: float
    num_initial_states: int
    num_rollouts_per_data_point: int
    save_every: int
    save_path: Union[str, Path]
    print_every: int = 1


class DatasetGenerator:
    """Generate a diffusion policy dataset for score function learning."""

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

        # Determine the size of the dataset that we'll hold in GPU memory
        # before saving out to disc.
        assert (
            langevin_options.num_noise_levels % datagen_options.save_every == 0
        ), "The number of noise levels must be divisible by the save frequency."

        # Save langevin sampling options, since we'll use them again when we
        # deploy the trained policy.
        save_path = Path(datagen_options.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "langevin_options.pkl", "wb") as f:
            pickle.dump(self.langevin_options, f)

        # Initialize the hdf5 file to save the dataset to
        self.h5_path = save_path / "dataset.h5"
        Y_shape = (prob.num_steps, prob.env.observation_size)
        U_shape = (prob.num_steps - 1, prob.env.action_size)
        with h5py.File(self.h5_path, "w") as f:
            f.create_dataset(
                "Y", (0, *Y_shape), maxshape=(None, *Y_shape), dtype="float32"
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
            f.create_dataset(
                "cost", (0, 1), maxshape=(None, 1), dtype="float32"
            )

    def estimate_noised_score(
        self,
        x0: State,
        controls: jnp.ndarray,
        sigma: float,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
            The cost J(U | x₀) of the control tape.
            The observation sequence [y₀, y₁, ..., y_T] associated with U.
        """
        M = self.datagen_options.num_rollouts_per_data_point
        lmbda = self.datagen_options.starting_temperature * sigma**2

        # Sample control tapes Ũʲ ~ 𝒩(U,σ²)
        rng, ctrl_rng = jax.random.split(rng)
        U_noised = controls + sigma * jax.random.normal(
            ctrl_rng, (M, *controls.shape)
        )

        # Include the mean in the rollouts. This technically changes the
        # distribution slightly, but it allows us log the cost and sequence
        # of observations without doing another rollout.
        U_noised = jnp.concatenate([controls[None], U_noised], axis=0)

        # Compute the cost of each control tape
        J, states = jax.vmap(self.prob.rollout, in_axes=(None, 0))(x0, U_noised)

        # Get the cost and state trajectory for the un-noised control tape
        cost = J[0]
        observations = states.obs[0]

        # Normalize costs for better numerics
        J = J - jnp.min(J, axis=0)

        # Compute importance weights
        weights = jnp.exp(-J / lmbda)
        weights = weights / (jnp.sum(weights, axis=0) + 1e-6)  # avoid / 0

        # Compute the noised score estimate
        deltaU = U_noised - controls
        score_estimate = jnp.einsum("i,i...->...", weights, deltaU)
        score_estimate /= sigma**2

        return score_estimate, cost, observations

    def save_dataset(self, dataset: DiffusionDataset) -> None:
        """Add a dataset to the hdf5 file.

        Args:
            dataset: The dataset to save.
        """
        # Write the dataset to the hdf5 file
        with h5py.File(self.h5_path, "a") as f:
            Y, U, s, k, sigma, cost = (
                f["Y"],
                f["U"],
                f["s"],
                f["k"],
                f["sigma"],
                f["cost"],
            )
            num_existing_data_points = Y.shape[0]
            num_new_data_points = dataset.Y.shape[0]
            new_size = num_existing_data_points + num_new_data_points

            # Resize datasets to accomodate new data
            Y.resize(new_size, axis=0)
            U.resize(new_size, axis=0)
            s.resize(new_size, axis=0)
            k.resize(new_size, axis=0)
            sigma.resize(new_size, axis=0)
            cost.resize(new_size, axis=0)

            # Write the new data
            Y[num_existing_data_points:] = dataset.Y
            U[num_existing_data_points:] = dataset.U
            s[num_existing_data_points:] = dataset.s
            k[num_existing_data_points:] = dataset.k
            sigma[num_existing_data_points:] = dataset.sigma
            cost[num_existing_data_points:] = dataset.cost

    def allocate_dataset(self) -> DiffusionDataset:
        """Initialize an empty diffusion dataset that we can add to later.

        Returns:
            A dataset filled with zeros.
        """
        M = self.datagen_options.save_every
        N = self.datagen_options.num_initial_states
        ny = self.prob.env.observation_size
        nu = self.prob.env.action_size
        T = self.prob.num_steps
        return DiffusionDataset(
            Y=jnp.zeros((M, N, T, ny)),
            U=jnp.zeros((M, N, T - 1, nu)),
            s=jnp.zeros((M, N, T - 1, nu)),
            k=jnp.zeros((M, N, 1), dtype=jnp.int32),
            sigma=jnp.zeros((M, N, 1)),
            cost=jnp.zeros((M, N, 1)),
        )

    def add_to_dataset(
        self,
        dataset: DiffusionDataset,
        observations: jnp.ndarray,
        controls: jnp.ndarray,
        score: jnp.ndarray,
        k: jnp.ndarray,
        sigma: jnp.ndarray,
        cost: jnp.ndarray,
        i: int,
    ) -> DiffusionDataset:
        """Add new data to the dataset in the i-th slot.

        Args:
            dataset: The existing dataset.
            observations: The observation sequence Y.
            controls: The control sequence U.
            score: The noised score estimate.
            k: The noise level index.
            sigma: The noise level.
            cost: The total cost J(U | y₀) of the rollout.
            i: The index of the dataset to update

        Returns:
            The updated dataset.
        """
        Y = dataset.Y.at[i].set(observations)
        U = dataset.U.at[i].set(controls)
        s = dataset.s.at[i].set(score)
        k = dataset.k.at[i].set(k)
        sigma = dataset.sigma.at[i].set(sigma)
        cost = dataset.cost.at[i].set(cost)
        return dataset.replace(Y=Y, U=U, s=s, k=k, sigma=sigma, cost=cost)

    def langevin_step(
        self,
        controls: jnp.ndarray,
        score: jnp.ndarray,
        sigma: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Perform a single Langevin step on the control tape.

            Uᵏ⁺¹ = Uᵏ + αs + β√(2α)ε,
            ε ~ N(0, I).

        Note that the step size α is scaled by the noise level σₖ, as
        recommended by Song and Ermon, "Generative Modeling by Estimating
        Gradients of the Data Distribution", NeurIPS 2019.

        Args:
            controls: The control tape U.
            score: The score estimate s.
            sigma: The noise level σₖ.
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

        # Some helper functions
        jit_reset = jax.jit(jax.vmap(self.prob.env.reset))
        jit_score = jax.jit(
            jax.vmap(self.estimate_noised_score, in_axes=(0, 0, None, 0))
        )
        jit_update = jax.jit(
            lambda dataset, y, u, s, k, sigma, cost, i: self.add_to_dataset(
                dataset,
                y,
                u,
                s,
                jnp.tile(k, (N, 1)),
                jnp.tile(sigma, (N, 1)),
                jnp.expand_dims(cost, -1),
                i,
            ),
            donate_argnums=(0,),  # in-place update of the dataset
        )
        jit_langevin_step = jax.jit(
            jax.vmap(self.langevin_step, in_axes=(0, 0, None, 0)),
            donate_argnums=(0,),  # in-place update of the control tape
        )

        # Set the initial state
        rng, state_rng = jax.random.split(rng)
        state_rng = jax.random.split(state_rng, N)
        x0 = jit_reset(state_rng)

        # Allocate the dataset
        dataset = self.allocate_dataset()

        # Sample inital control tape U ~ 𝒩(0, σ_L²)
        rng, init_rng = jax.random.split(rng)
        U = self.langevin_options.starting_noise_level * jax.random.normal(
            init_rng, (N, self.prob.num_steps - 1, self.prob.env.action_size)
        )

        i = 0  # counter for which row of the dataset we're writing to
        for k in range(L - 1, -1, -1):
            rng, score_rng, step_rng = jax.random.split(rng, 3)

            # Set the noise level σₖ
            t = (L - k) / L
            sigma = self.langevin_options.starting_noise_level * jnp.exp(
                -self.langevin_options.noise_decay_rate * t
            )

            # Compute the score estimate
            score_rngs = jax.random.split(score_rng, N)
            s, cost, Y = jit_score(x0, U, sigma, score_rngs)

            # Update the dataset
            dataset = jit_update(dataset, Y, U, s, k, sigma, cost, i)
            i += 1

            # Update the control tape from the previous score
            step_rngs = jax.random.split(step_rng, N)
            U = jit_langevin_step(U, s, sigma, step_rngs)

            if k % self.datagen_options.print_every == 0:
                print(
                    f"k = {k}, σₖ = {sigma:.4f}, "
                    f"cost = {jnp.mean(cost):.4f} +/- {jnp.std(cost):.4f}, "
                    f"time = {datetime.now() - start_time}"
                )

            if k % self.datagen_options.save_every == 0:
                # Flatten the dataset for saving
                flat_dataset = jax.tree.map(
                    lambda x: jnp.reshape(x, (-1, *x.shape[2:])), dataset
                )
                self.save_dataset(flat_dataset)
                i = 0
