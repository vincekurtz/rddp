from flax.struct import dataclass

from rddp.tasks.base import OptimalControlProblem


@dataclass
class AnnealedLangevinOptions:
    """Parameters for annealed Langevin dynamics.

    Annealed Langevin dynamics samples from the target distribution

        p(U | x₀) ∝ exp(-J(U | x₀) / λ),

    by considering intermediate noised distributions

        pₖ(U | x₀) = ∫ p(Ũ | x₀)N(Ũ;U,σₖ²)dŨ

    with a geometrically decreasing sequence of noise levels k = L, L-1, ..., 0.

    Attributes:
        temperature: The temperature λ
        num_noise_levels: The number of noise levels L.
        starting_noise_level: The starting noise level σ_L.
        noise_decay_rate: The noise decay rate σₖ₋₁ = γ σₖ.
    """

    temperature: float
    num_noise_levels: int
    starting_noise_level: int
    noise_decay_rate: float


@dataclass
class DatasetGenerationOptions:
    """Parameters for generating a diffusion policy dataset.

    Attributes:
        num_initial_states: The number of initial states x₀ to sample.
        num_data_points: The number of data points per initial state, N.
        num_rollouts: The number of rollouts used to estimate each score, M.
    """

    num_initial_states: int
    num_data_points: int
    num_rollouts: int


class DatasetGenerator:
    """Generate a diffusion policy dataset for score function learning.

    The dataset consists of tuples
        (x₀, U, ŝ, k),
    where
        x₀ is the initial state,
        U is the control sequence U = [u₀, u₁, ..., u_T₋₁],
        ŝ is the noised score estimate ŝ = σₖ² ∇ log pₖ(U | x₀),
        and k is noise level index.
    """

    def __init__(
        self,
        prob: OptimalControlProblem,
        langevin_options: AnnealedLangevinOptions,
        datagen_options: DatasetGenerationOptions,
    ):
        """Initialize the dataset generator.

        Args:
            prob: The optimal control problem defining the cost J(U | x₀).
            langevin_options: Sampling (e.g., temperature) settings.
            datagen_options: Dataset generation (e.g., num rollouts) settings.
        """
        self.prob = prob
        self.langevin_options = langevin_options
        self.datagen_options = datagen_options
