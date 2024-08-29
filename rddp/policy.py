import pickle
from pathlib import Path
from typing import Any, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from rddp.utils import AnnealedLangevinOptions, annealed_langevin_sample

Params = Any


class DiffusionPolicy:
    """A helper object that stores a trained diffusion policy."""

    def __init__(
        self,
        net: nn.Module,
        params: Params,
        options: AnnealedLangevinOptions,
        action_shape: Tuple,
    ):
        """Initialize the policy object.

        Args:
            net: The score network architecture, s_θ(y, U, σ).
            params: The trained network parameters parameters θ.
            options: The annealed Langevin sampling options.
            action_shape: The shape of the action sequence, (num_steps - 1, nu).
        """
        self.net = net
        self.params = params
        self.options = options
        self.action_shape = action_shape

    def save(self, path: Union[str, Path]) -> None:
        """Save the policy to a file.

        Args:
            path: The path to save the policy to.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "DiffusionPolicy":
        """Load the policy from a file.

        Args:
            path: The path to load the policy from.

        Returns:
            The loaded policy object.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def apply(self, y0: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Deploy the trained policy to generate optimal actions.

        Args:
            y0: The initial observation.
            rng: The random number generator.

        Returns:
            The optimal action sequence.
        """
        # Guess an initial control sequence
        rng, guess_rng = jax.random.split(rng, 2)
        U_guess = self.options.starting_noise_level * jax.random.normal(
            guess_rng, self.action_shape
        )

        # Do annealed langevin sampling
        rng, langevin_rng = jax.random.split(rng)
        U, _ = annealed_langevin_sample(
            options=self.options,
            y0=y0,
            controls=U_guess,
            score_fn=lambda y, u, sigma, rng: self.net.apply(
                self.params, y, u, jnp.atleast_1d(sigma)
            ),
            rng=langevin_rng,
        )

        return U
