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


def sample_target_distribution(
    rng: jax.random.PRNGKey, num_samples: int
) -> jnp.ndarray:
    """Sample from the target distribution using rejection sampling."""
    # Proposal distribution is a mixture of two Gaussians
    rng, p1_rng, p2_rng, choice_rng = jax.random.split(rng, 4)
    p1 = 0.3 * jax.random.normal(p1_rng, (num_samples,)) + 1.0
    p2 = 0.3 * jax.random.normal(p2_rng, (num_samples,)) - 1.0
    choice = jax.random.bernoulli(choice_rng, 0.8, (num_samples,))
    x_proposed = jnp.where(choice, p1, p2)

    # Acceptance ratio
    q = (
        jax.scipy.stats.norm.pdf(x_proposed, loc=1.0, scale=0.3) * 0.8
        + jax.scipy.stats.norm.pdf(x_proposed, loc=-1.0, scale=0.3) * 0.2
    )
    p = target_distribution(x_proposed)
    alpha = p / (2 * q)  # Extra fudge factor to make sure q > p

    # Accept or reject
    rng, accept_rng = jax.random.split(rng)
    accept = jax.random.uniform(accept_rng, (num_samples,)) <= alpha

    x = jnp.where(accept, x_proposed, jnp.nan)
    x = x[~jnp.isnan(x)]

    return x


def true_score(x: jnp.ndarray) -> jnp.ndarray:
    """The true score of the target distribution."""
    log_p = lambda x: jnp.log(target_distribution(x))
    grad_log_p = jax.grad(log_p)
    return jax.vmap(grad_log_p)(x)


def energy_gradient(x: jnp.ndarray) -> jnp.ndarray:
    """The gradient of the energy function."""
    grad_energy = jax.vmap(jax.grad(energy))
    return grad_energy(x)


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

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 15))

    ax[0].set_title("Energy")
    ax[0].plot(x, energy(x), label="Energy")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("E(x)")

    ax[1].set_title("Target Distrubtion")
    ax[1].plot(x, target_distribution(x), label="Target distribution")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("p(x)")

    ax[2].set_title("Target Distribution Samples")
    rng, sample_rng = jax.random.split(rng)
    samples = sample_target_distribution(sample_rng, 5000)
    ax[2].hist(samples, bins=30)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("occurances")

    ax[3].set_title("Score")
    ax[3].plot(x, true_score(x), label="True score")

    x = jnp.linspace(-3, 3, 100)
    vmap_noised_score = jax.vmap(
        estimate_noised_score, in_axes=(0, None, None, 0)
    )
    rng, sample_rng = jax.random.split(rng)
    sample_rng = jax.random.split(rng, x.shape[0])
    s_hat = vmap_noised_score(x, 50, 0.01, sample_rng)
    ax[3].scatter(x, s_hat, alpha=0.5, label="Estimated, sigma=0.01")

    rng, sample_rng = jax.random.split(rng)
    sample_rng = jax.random.split(rng, x.shape[0])
    s_hat = vmap_noised_score(x, 50, 0.1, sample_rng)
    ax[3].scatter(x, s_hat, alpha=0.5, label="Estimated, sigma=0.1")

    ax[3].set_xlabel("x")
    ax[3].set_ylabel("s(x)")
    ax[3].legend()

    plt.show()
