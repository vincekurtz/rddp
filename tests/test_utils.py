import jax
import jax.numpy as jnp

from rddp.utils import (
    AnnealedLangevinOptions,
    DiffusionDataset,
    annealed_langevin_sample,
)


def test_annealed_langevin_sample() -> None:
    """Do annealed langevin sampling on a simple toy example."""
    rng = jax.random.PRNGKey(0)
    u_nom = jnp.array([4.3, -2.6])

    def cost_fn(u: jnp.ndarray) -> jnp.ndarray:
        """A simple cost function."""
        return jnp.sum(jnp.square(u - u_nom))

    def score_fn(
        y: jnp.ndarray, u: jnp.ndarray, sigma: float, rng: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """A noised score function estimate based on the MPPI update rule."""
        rng, sample_rng = jax.random.split(rng)
        u_noised = u + sigma * jax.random.normal(sample_rng, (16, *u.shape))

        J = jax.vmap(cost_fn)(u_noised)
        J = J - jnp.min(J)
        weights = jnp.exp(-J / 0.001)
        weights = weights / jnp.sum(weights)

        return weights.dot(u_noised - u) / sigma**2

    # First do regular Langevin sampling. We should get close to u_nom.
    options = AnnealedLangevinOptions(
        denoising_steps=5000,
        starting_noise_level=1.0,
        step_size=0.01,
        noise_injection_level=1.0,
    )

    rng, langevin_rng = jax.random.split(rng)
    U, data = annealed_langevin_sample(
        options=options,
        y0=jnp.zeros(3),  # unused in this example
        controls=jnp.zeros(2),
        score_fn=score_fn,
        rng=langevin_rng,
    )

    assert isinstance(data, DiffusionDataset)
    assert data.U.shape == (options.denoising_steps, 2)
    assert data.s.shape == data.U.shape
    assert data.sigma.shape == (options.denoising_steps, 1)
    assert data.k.shape == (options.denoising_steps, 1)

    cost = cost_fn(U)
    assert cost < 0.1

    # Now do it again without noise injection. This is gradient descent rather
    # than Langevin dynamics, so we should get much closer to u_nom.
    options = options.replace(noise_injection_level=0.0)
    U, data = annealed_langevin_sample(
        options=options,
        y0=jnp.zeros(3),
        controls=jnp.zeros(2),
        score_fn=score_fn,
        rng=langevin_rng,
    )

    zero_noise_cost = cost_fn(U)
    assert zero_noise_cost < 1e-4
    assert zero_noise_cost < cost


if __name__ == "__main__":
    test_annealed_langevin_sample()
