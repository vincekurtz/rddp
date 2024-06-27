import jax

from rddp.architectures import DiffusionPolicyMLP


def test_mlp() -> None:
    """Test the basic diffusion policy MLP architecture."""
    rng = jax.random.PRNGKey(0)
    net = DiffusionPolicyMLP(layer_sizes=[32, 32])

    # Fake data for initialization
    rng, x_rng, u_rng, sigma_rng = jax.random.split(rng, 4)
    x0 = jax.random.normal(x_rng, (2,))
    U = jax.random.normal(u_rng, (19, 2))
    sigma = jax.random.normal(sigma_rng, (1,))

    # Forward pass with fake data
    rng, params_rng = jax.random.split(rng)
    params = net.init(params_rng, x0, U, sigma)
    s = net.apply(params, x0, U, sigma)
    assert s.shape == U.shape

    # Forward pass with extra (e.g., batch) dimensions
    rng, x_rng, u_rng, sigma_rng = jax.random.split(rng, 4)
    x0 = jax.random.normal(x_rng, (3, 4, 2))
    U = jax.random.normal(u_rng, (3, 4, 19, 2))
    sigma = jax.random.normal(sigma_rng, (3, 4, 1))
    s = net.apply(params, x0, U, sigma)
    assert s.shape == U.shape


if __name__ == "__main__":
    test_mlp()
