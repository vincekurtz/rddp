from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp


class DynamicalSystem(ABC):
    """A discrete-time dynamical system.

        xₜ₊₁ = f(xₜ, uₜ),
        yₜ = g(xₜ),

    where x is the state, u is the control action, and y is the output.
    """

    @abstractmethod
    def f(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Forward dynamics function.

        Args:
            x: The current state xₜ.
            u: The control action uₜ to apply.

        Returns:
            The next state xₜ₊₁.
        """

    @abstractmethod
    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Output function.

        Args:
            x: The current state.

        Returns:
            The output measurement.
        """

    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """The shape of the state space."""

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """The shape of the action space."""

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """The shape of the observation space."""
