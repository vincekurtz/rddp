# Reward-Driven Diffusion Policy

Training diffusion policies without demonstrations via connections between
Langevin sampling and path integral control.

## Setup (Conda)

Set up a conda env with Cuda 12.3 support (first time only):

```bash
conda env create -n [env_name] -f environment.yml
```

Enter the conda env:

```bash
conda activate [env_name]
```

Install dependencies:

```bash
pip install -e . --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Set up pre-commit hooks:

```bash
pre-commit autoupdate
pre-commit install
```

## Usage

Run unit tests:

```bash
pytest
```

Other demos can be found in the `examples` folder.
