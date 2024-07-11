import h5py
import jax
import jax.numpy as jnp

from rddp.utils import DiffusionDataset


def generate_random_dataset(
    rng: jax.random.PRNGKey, num_samples: int
) -> DiffusionDataset:
    """Generate a toy dataset."""
    rng, x0_rng, U_rng, s_rng, sigma_rng, k_rng = jax.random.split(rng, 6)
    x0 = jax.random.uniform(x0_rng, (num_samples, 2))
    U = jax.random.uniform(U_rng, (num_samples, 49, 2))
    s = jax.random.uniform(s_rng, (num_samples, 49, 2))
    sigma = jax.random.uniform(sigma_rng, (num_samples, 1))
    k = jax.random.uniform(k_rng, (num_samples, 1))
    return DiffusionDataset(x0=x0, U=U, s=s, sigma=sigma, k=k)


def save_toy_dataset(num_samples: int = 1000) -> None:
    """Save a toy dataset to an hdf5 file."""
    rng = jax.random.PRNGKey(0)

    # Create an empty hdf5 file
    with h5py.File("toy_dataset.h5", "w") as f:
        f.create_dataset("x0", (0, 2), maxshape=(None, 2), dtype="float32")
        f.create_dataset(
            "U", (0, 49, 2), maxshape=(None, 49, 2), dtype="float32"
        )
        f.create_dataset(
            "s", (0, 49, 2), maxshape=(None, 49, 2), dtype="float32"
        )
        f.create_dataset("sigma", (0, 1), maxshape=(None, 1), dtype="float32")
        f.create_dataset("k", (0, 1), maxshape=(None, 1), dtype="float32")

    jit_generate = jax.jit(
        lambda rng: generate_random_dataset(rng, num_samples)
    )

    for i in range(1000):
        if i % 100 == 0:
            print(f"Generating dataset {i}...")

        # Generate a random dataset
        rng, gen_rng = jax.random.split(rng)
        dataset = jit_generate(gen_rng)

        # Save the dataset to the hdf5 file
        with h5py.File("toy_dataset.h5", "a") as f:
            x0_data = f["x0"]
            U_data = f["U"]
            s_data = f["s"]
            sigma_data = f["sigma"]
            k_data = f["k"]

            # Resize datasets to accomodate new data
            num_existing_data_points = x0_data.shape[0]
            num_new_data_points = dataset.x0.shape[0]
            new_size = num_existing_data_points + num_new_data_points
            x0_data.resize(new_size, axis=0)
            U_data.resize(new_size, axis=0)
            s_data.resize(new_size, axis=0)
            sigma_data.resize(new_size, axis=0)
            k_data.resize(new_size, axis=0)

            # Write new data
            x0_data[num_existing_data_points:] = dataset.x0
            U_data[num_existing_data_points:] = dataset.U
            s_data[num_existing_data_points:] = dataset.s
            sigma_data[num_existing_data_points:] = dataset.sigma
            k_data[num_existing_data_points:] = dataset.k


def load_toy_dataset() -> None:
    """Load the toy dataset from the hdf5 file."""
    with h5py.File("toy_dataset.h5", "r") as f:
        x0 = jnp.array(f["x0"][0:1000])
        U = jnp.array(f["U"][0:1000])
        s = jnp.array(f["s"][0:1000])
        sigma = jnp.array(f["sigma"][0:1000])
        k = jnp.array(f["k"][0:1000])
    dataset = DiffusionDataset(x0=x0, U=U, s=s, sigma=sigma, k=k)
    print(dataset.x0.shape)
    print(type(dataset.x0))
    new_x = dataset.x0 + 1
    print(new_x.devices())


if __name__ == "__main__":
    # save_toy_dataset(10_000)
    load_toy_dataset()
