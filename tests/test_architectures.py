import jax

from rddp.architectures import ScoreMLP


def test_mlp() -> None:
    """Test the basic diffusion policy MLP architecture."""
    rng = jax.random.PRNGKey(0)
    net = ScoreMLP(layer_sizes=[32, 32])

    # Fake data for initialization
    rng, y_rng, u_rng, sigma_rng = jax.random.split(rng, 4)
    y0 = jax.random.normal(y_rng, (2,))
    U = jax.random.normal(u_rng, (19, 2))
    sigma = jax.random.normal(sigma_rng, (1,))

    # Forward pass with fake data
    rng, params_rng = jax.random.split(rng)
    params = net.init(params_rng, y0, U, sigma)
    s = net.apply(params, y0, U, sigma)
    assert s.shape == U.shape

    # Forward pass with extra (e.g., batch) dimensions
    rng, y_rng, u_rng, sigma_rng = jax.random.split(rng, 4)
    y0 = jax.random.normal(y_rng, (3, 4, 2))
    U = jax.random.normal(u_rng, (3, 4, 19, 2))
    sigma = jax.random.normal(sigma_rng, (3, 4, 1))
    s = net.apply(params, y0, U, sigma)
    assert s.shape == U.shape


if __name__ == "__main__":
    test_mlp()
