import pickle
from pathlib import Path
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, Dataset, default_collate

from rddp.utils import DiffusionData


class TorchDiffusionDataset(Dataset):
    """A custom pytorch dataset for loading saved training data."""

    def __init__(self, save_path: Union[str, Path]):
        """Initialize the dataset.

        The dataset is saved by the DatasetGenerator class to a directory
        with the following structure:
            save_path/
                diffusion_data_0.pkl
                diffusion_data_1.pkl
                ...
                diffusion_data_N.pkl
                langevin_options.pkl

        Each pickle file contains a DiffusionData object with

        Args:
            save_path: The path to the saved dataset.
        """
        self.save_path = Path(save_path)

        # Load the data from the saved files
        # TODO: this loads all the data into RAM, which may not be feasible
        # for large datasets. Consider loading the data lazily.
        self.data_points_per_set = None
        self.sets = []
        i = 0
        for p in self.save_path.iterdir():
            if p.suffix == ".pkl" and "langevin_options" not in p.stem:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                    assert isinstance(data, DiffusionData)
                    self.sets.append(data)

                    if self.data_points_per_set is None:
                        self.data_points_per_set = len(data.x0)
                    else:
                        assert len(data.x0) == self.data_points_per_set

                i += 1

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return self.data_points_per_set * len(self.sets)

    def __getitem__(self, idx: int) -> DiffusionData:
        """Return the idx-th data point in the dataset."""
        set_idx, data_idx = divmod(idx, self.data_points_per_set)
        data = jax.tree.map(lambda x: x[data_idx], self.sets[set_idx])

        x0 = np.array(data.x0, dtype=jnp.float32)
        U = np.array(data.U, dtype=jnp.float32)
        s = np.array(data.s, dtype=jnp.float32)
        sigma = np.array(data.sigma, dtype=jnp.float32)
        k = np.array(data.k, dtype=jnp.float32)

        return {"x0": x0, "U": U, "s": s, "sigma": sigma, "k": k}


class TorchDiffusionDataLoader(DataLoader):
    """Pytorch data loader that returns jax numpy arrays."""

    def __init__(self, dataset: Dataset, **kwargs: Any):
        """Initialize the data loader."""
        super().__init__(
            dataset,
            collate_fn=lambda batch: jax.tree_util.tree_map(
                jnp.asarray, default_collate(batch)
            ),
            **kwargs,
        )
