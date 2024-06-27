from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class DiffusionPolicyMLP(nn.Module):
    """A simple fully-connected network for diffusion policy.

    This network estimates the noised score

           s_θ(x₀, U, σₖ) ≈ ∇ log pₖ(U | x₀)

    where U = [u₀, u₁, ..., u_T₋₁] is the control sequence, x₀ is the initial
    state, and k is the noise level index.

    Attributes:
        hidden_layer_sizes: The number of units in each hidden layer.
        activate_final: Whether to apply an activation to the final output.
        bias: Whether to include bias terms in the network.
    """

    layer_sizes: Sequence[int]
    bias: bool = True

    @nn.compact
    def __call__(
        self,
        start_state: jnp.ndarray,
        control_tape: jnp.ndarray,
        sigma: jnp.ndarray,
    ):
        """Forward pass to estimate the score of a given action sequence.

        Args:
            start_state: The initial state x₀.
            control_tape: The control sequence U = [u₀, u₁, ..., u_T₋₁].
            sigma: The noise level σₖ.

        Returns:
            The control sequence U = [u₀, u₁, ..., u_T₋₁].
        """
        # Flatten all inputs into a single vector
        extra_dims = sigma.shape[:-1]
        flat_control_tape = control_tape.reshape((*extra_dims, -1))
        x = jnp.concatenate([start_state, flat_control_tape, sigma], axis=-1)

        for size in self.layer_sizes:
            x = nn.Dense(size, use_bias=self.bias)(x)
            x = nn.swish(x)
        x = nn.Dense(flat_control_tape.shape[-1], use_bias=self.bias)(x)

        # Reshape the output to match the control tape shape
        x = x.reshape(*control_tape.shape)

        return x
