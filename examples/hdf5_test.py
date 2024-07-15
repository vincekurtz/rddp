import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

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
    with h5py.File("/tmp/toy_dataset.h5", "w") as f:
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
        with h5py.File("/tmp/toy_dataset.h5", "a") as f:
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


def load_partial_dataset() -> None:
    """Load part of the toy dataset into GPU memory."""
    for i in range(10):
        with h5py.File("/tmp/toy_dataset.h5", "r") as f:
            x0 = jnp.array(f["x0"][i : i + 1000])
            U = jnp.array(f["U"][i : i + 1000])
            s = jnp.array(f["s"][i : i + 1000])
            sigma = jnp.array(f["sigma"][i : i + 1000])
            k = jnp.array(f["k"][i : i + 1000])
        dataset = DiffusionDataset(x0=x0, U=U, s=s, sigma=sigma, k=k)
        print(dataset.x0.shape)
        print(type(dataset.x0))
        new_x = dataset.x0 + 1
        print(new_x.devices())


@jax.jit
def do_toy_computation_on_batch(dataset: DiffusionDataset) -> jnp.ndarray:
    """Do some toy computation on a batch of data, to imitate training."""
    res = dataset.U + dataset.s
    res = jnp.sum(res, axis=(1, 2))
    res = res * dataset.x0[:, 0] + dataset.x0[:, 1]
    return res.mean()


def gpu_only(batch_size=2048) -> None:
    """Do the toy computation on the full dataset, loading it all to GPU."""
    rng = jax.random.PRNGKey(0)

    st = time.time()
    with h5py.File("/tmp/toy_dataset.h5", "r") as f:
        num_data_points = f["x0"].shape[0]
        x0 = np.array(f["x0"])
        U = np.array(f["U"])
        s = np.array(f["s"])
        sigma = np.array(f["sigma"])
        k = np.array(f["k"])
    print(f"Loaded {num_data_points} samples in {time.time() - st:.2f} seconds")

    num_batches = num_data_points // batch_size
    rng, shuffle_rng = jax.random.split(rng)
    perm = jax.random.permutation(shuffle_rng, num_data_points)
    

    st = time.time() 
    dataset = DiffusionDataset(
        x0=jnp.array(x0),
        U=jnp.array(U),
        s=jnp.array(s),
        k=jnp.array(k),
        sigma=jnp.array(sigma),
    )
    print("Moved to GPU in", time.time() - st)

    def scan_fn(carry, batch):
        # idx = perm[batch * batch_size : (batch + 1) * batch_size]
        idx = jax.lax.dynamic_slice_in_dim(perm, batch * batch_size, batch_size)
        batch_dataset = jax.tree.map(lambda x: x[idx], dataset)
        res = do_toy_computation_on_batch(batch_dataset)
        return carry + res, res

    st = time.time()
    res, _ = jax.lax.scan(scan_fn, 0.0, jnp.arange(num_batches))

    print(f"Epoch time: {time.time() - st:.4f} seconds")
    print(res)


def disc_to_gpu(batch_size=2048) -> None:
    """Do the toy computation on the full dataset, loading each batch to GPU."""
    rng = jax.random.PRNGKey(0)
    with h5py.File("/tmp/toy_dataset.h5", "r") as f:
        num_data_points = f["x0"].shape[0]
        x0, U, s, sigma, k = f["x0"], f["U"], f["s"], f["sigma"], f["k"]

        num_batches = num_data_points // batch_size
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, num_data_points)

        for batch in range(num_batches):
            print(f"Batch {batch + 1}/{num_batches}")
            st = time.time()
            idx = jnp.sort(perm[batch * batch_size : (batch + 1) * batch_size])
            x0_batch = jnp.array(x0[idx])
            U_batch = jnp.array(U[idx])
            s_batch = jnp.array(s[idx])
            sigma_batch = jnp.array(sigma[idx])
            k_batch = jnp.array(k[idx])
            dataset = DiffusionDataset(
                x0=x0_batch, U=U_batch, s=s_batch, sigma=sigma_batch, k=k_batch
            )
            print(f"  Loaded in {time.time() - st:.2f} seconds")
            assert dataset.x0.shape == (batch_size, 2)

            st = time.time()
            res = do_toy_computation_on_batch(dataset)
            print(f"  Computation took {time.time() - st:.2f} seconds")


def disc_to_ram_to_gpu(batch_size=2048) -> None:
    """Do the toy computation on the full dataset, loading to RAM first."""
    rng = jax.random.PRNGKey(0)

    # Load the dataset to RAM
    st = time.time()
    with h5py.File("/tmp/toy_dataset.h5", "r") as f:
        x0, U, s, sigma, k = f["x0"], f["U"], f["s"], f["sigma"], f["k"]
        x0 = np.array(x0)
        U = np.array(U)
        s = np.array(s)
        sigma = np.array(sigma)
        k = np.array(k)
    print(f"Loaded to RAM in {time.time() - st:.2f} seconds")

    num_data_points = x0.shape[0]
    num_batches = num_data_points // batch_size
    rng, shuffle_rng = jax.random.split(rng)
    perm = jax.random.permutation(shuffle_rng, num_data_points)

    # Do the actual computation, shifting batches to GPU as needed
    total_start_time = time.time()
    total = 0.0
    for batch in range(num_batches):
        print(f"Batch {batch + 1}/{num_batches}")
        st = time.time()
        idx = perm[batch * batch_size : (batch + 1) * batch_size]
        x0_batch = jnp.array(x0[idx])
        U_batch = jnp.array(U[idx])
        s_batch = jnp.array(s[idx])
        sigma_batch = jnp.array(sigma[idx])
        k_batch = jnp.array(k[idx])
        dataset = DiffusionDataset(
            x0=x0_batch, U=U_batch, s=s_batch, sigma=sigma_batch, k=k_batch
        )
        print(f"  Loaded in {time.time() - st:.5f} seconds")
        assert dataset.x0.shape == (batch_size, 2)

        st = time.time()
        res = do_toy_computation_on_batch(dataset)
        print(f"  Computation took {time.time() - st:.5f} seconds")
        total += res

    print(f"Epoch time: {time.time() - total_start_time:.2f} seconds")
    print(total)


if __name__ == "__main__":
    # save_toy_dataset(7_000)
    # load_full_dataset()
    # disc_to_gpu()
    #disc_to_ram_to_gpu(batch_size=4096)
    gpu_only(batch_size=4096)
