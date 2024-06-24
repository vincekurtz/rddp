import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

"""
A simple 1D example of score estimation via path-integral-style sampling.
"""

# Global path integral temperature
LAMBDA = 0.1


def energy(x: jnp.ndarray) -> jnp.ndarray:
    """The energy/cost function we want to minimize."""
    return -jnp.log(
        jnp.exp(-2 * (x - 1) ** 2) + 0.9 * jnp.exp(-2 * (x + 1) ** 2)
    )


def target_distribution(x: jnp.ndarray) -> jnp.ndarray:
    """The target distribution p(x) we want to sample from."""
    return jnp.exp(-energy(x) / LAMBDA)


def true_score(x: jnp.ndarray) -> jnp.ndarray:
    """The true score of the target distribution."""
    log_p = lambda x: jnp.log(target_distribution(x))
    grad_log_p = jax.grad(log_p)
    return jax.vmap(grad_log_p)(x)


def estimate_noised_score(
    x: jnp.ndarray, num_samples: int, sigma: float, rng: jax.random.PRNGKey
) -> jnp.ndarray:
    """Estimate the noised score at x using path-integral-style sampling."""
    x_i = jax.random.normal(rng, (num_samples,)) * sigma + x
    J = energy(x_i)
    J = J - jnp.min(J)
    weights = jnp.exp(-J / LAMBDA)
    weights = weights / jnp.sum(weights)
    s_hat = weights.dot(x_i - x)
    return s_hat / sigma**2


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    x = jnp.linspace(-3, 3, 1000)

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.title("Energy")
    plt.plot(x, energy(x), label="Energy")
    plt.xlabel("x")
    plt.ylabel("E(x)")

    plt.subplot(3, 1, 2)
    plt.title("Target Distrubtion")
    plt.plot(x, target_distribution(x), label="Target distribution")
    plt.xlabel("x")
    plt.ylabel("p(x)")

    plt.subplot(3, 1, 3)
    plt.title("Score")
    plt.plot(x, true_score(x), label="True score")

    x = jnp.linspace(-3, 3, 100)
    vmap_noised_score = jax.vmap(
        estimate_noised_score, in_axes=(0, None, None, 0)
    )
    rng, sample_rng = jax.random.split(rng)
    sample_rng = jax.random.split(rng, x.shape[0])
    s_hat = vmap_noised_score(x, 50, 0.01, sample_rng)
    plt.scatter(x, s_hat, alpha=0.5, label="Estimated, sigma=0.01")

    rng, sample_rng = jax.random.split(rng)
    sample_rng = jax.random.split(rng, x.shape[0])
    s_hat = vmap_noised_score(x, 50, 0.1, sample_rng)
    plt.scatter(x, s_hat, alpha=0.5, label="Estimated, sigma=0.1")

    plt.xlabel("x")
    plt.ylabel("s(x)")
    plt.legend()

    plt.show()
