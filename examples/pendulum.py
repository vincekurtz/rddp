import pickle
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rddp.architectures import ScoreMLP
from rddp.envs.pendulum import PendulumEnv
from rddp.generation import DatasetGenerationOptions, DatasetGenerator
from rddp.gradient_descent import solve as solve_gd
from rddp.ocp import OptimalControlProblem
from rddp.training import TrainingOptions, train
from rddp.utils import AnnealedLangevinOptions, annealed_langevin_sample

# Global planning horizon definition
HORIZON = 20


def solve_with_gradient_descent() -> None:
    """Solve the optimal control problem using simple gradient descent."""
    rng = jax.random.PRNGKey(0)
    prob = OptimalControlProblem(PendulumEnv(HORIZON), HORIZON)

    rng, reset_rng = jax.random.split(rng)
    x0 = prob.env.reset(reset_rng)
    U, _, _ = solve_gd(prob, x0, max_iter=5000, print_every=500)

    prob.env.plot_scenario()
    _, states = prob.rollout(x0, U)
    plt.plot(states.obs[:, 0], states.obs[:, 1], "o-")
    plt.show()


def generate_dataset(plot: bool = True) -> None:
    """Generate some training data."""
    rng = jax.random.PRNGKey(0)
    save_path = "/tmp/pendulum"

    # Problem setup
    prob = OptimalControlProblem(PendulumEnv(HORIZON), HORIZON)
    langevin_options = AnnealedLangevinOptions(
        num_noise_levels=500,
        starting_noise_level=1.0,
        num_steps=100,
        step_size=0.001,
        noise_injection_level=0.0,
    )
    gen_options = DatasetGenerationOptions(
        starting_temperature=1.0,
        num_initial_states=1,
        num_rollouts_per_data_point=1024,
        save_every=100,
        save_path=save_path,
    )
    generator = DatasetGenerator(prob, langevin_options, gen_options)

    st = time.time()
    rng, gen_rng = jax.random.split(rng)
    generator.generate_and_save(gen_rng)
    print(f"Data generation took {time.time() - st:.2f} seconds")


def fit_score_model() -> None:
    """Fit a simple score model to the generated data."""
    # Specify location of the training data
    data_dir = "/tmp/pendulum/"

    # Load the langiven sampling options
    with open(data_dir + "langevin_options.pkl", "rb") as f:
        langevin_options = pickle.load(f)

    # Set up the training options and the score network
    training_options = TrainingOptions(
        batch_size=5120,
        num_superbatches=1,
        epochs=50,
        learning_rate=1e-3,
    )
    net = ScoreMLP(layer_sizes=(128,) * 3)

    # Train the score network
    st = time.time()
    params, metrics = train(net, data_dir + "dataset.h5", training_options)
    print(f"Training took {time.time() - st:.2f} seconds")

    # Save the trained model and parameters
    fname = "/tmp/pendulum_score_model.pkl"
    with open(fname, "wb") as f:
        data = {
            "params": params,
            "net": net,
            "langevin_options": langevin_options,
        }
        pickle.dump(data, f)
    print(f"Saved trained model to {fname}")


def deploy_trained_model(
    plot: bool = True, animate: bool = False, save_path: str = None
) -> None:
    """Use the trained model to generate optimal actions."""
    rng = jax.random.PRNGKey(0)
    prob = OptimalControlProblem(PendulumEnv(HORIZON), num_steps=HORIZON)

    def rollout_from_obs(y0: jnp.ndarray, u: jnp.ndarray):
        """Do a rollout from an observation, and return observations."""
        x0 = prob.env.reset(rng)
        x0 = x0.tree_replace(
            {
                "pipeline_state.q": jnp.array([y0[0]]),
                "pipeline_state.qd": jnp.array([y0[1]]),
                "obs": y0,
            }
        )
        cost, X = prob.rollout(x0, u)
        return cost, X.obs

    # Load the trained score network
    with open("/tmp/pendulum_score_model.pkl", "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    net = data["net"]
    options = data["langevin_options"]

    # Decide how much noise to add in the Langevin sampling
    options = options.replace(noise_injection_level=0.0)

    def optimize_control_tape(rng: jax.random.PRNGKey):
        """Optimize the control sequence using Langevin dynamics."""
        # Guess an initial control sequence
        rng, guess_rng = jax.random.split(rng, 2)
        U_guess = options.starting_noise_level * jax.random.normal(
            guess_rng, (prob.num_steps - 1, 1)
        )

        # Set the initial state and observation
        rng, state_rng = jax.random.split(rng)
        x0 = prob.env.reset(state_rng)

        # Do annealed langevin sampling
        rng, langevin_rng = jax.random.split(rng)
        U, data = annealed_langevin_sample(
            options=options,
            y0=x0.obs,
            u_init=U_guess,
            score_fn=lambda x, u, sigma, rng: net.apply(
                params, x, u, jnp.array([sigma])
            ),
            rng=langevin_rng,
        )

        return U, data

    # Optimize from a bunch of initial guesses
    num_samples = 32
    rng, opt_rng = jax.random.split(rng)
    opt_rng = jax.random.split(opt_rng, num_samples)
    st = time.time()
    _, data = jax.vmap(optimize_control_tape)(opt_rng)
    print(f"Sample generation took {time.time() - st:.2f} seconds")

    y0 = data.y0[:, :, -1, :]  # take the last sample at each noise level
    U = data.U[:, :, -1, :]
    sigma = data.sigma[:, :, -1]
    costs, Xs = jax.vmap(jax.vmap(rollout_from_obs))(y0, U)
    print(f"Cost: {jnp.mean(costs[-1]):.4f} +/- {jnp.std(costs[-1]):.4f}")

    # Plot the sampled trajectories
    if plot:
        prob.env.plot_scenario()
        for i in range(num_samples):
            plt.plot(
                Xs[i, -1, :, 0], Xs[i, -1, :, 1], "o-", color="blue", alpha=0.5
            )
        plt.show()

    # Animate the trajectory generation process
    if animate:
        fig, ax = plt.subplots()
        prob.env.plot_scenario()
        path = ax.plot([], [], "o-")[0]

        def update(i: int):
            j, i = divmod(i, options.num_noise_levels)
            j = j % num_samples
            ax.set_title(f"σₖ={sigma[0, i, 0]:.4f}")
            path.set_data(Xs[j, i, :, 0], Xs[j, i, :, 1])
            return path

        anim = FuncAnimation(  # noqa: F841 anim must stay in scope until plot
            fig,
            update,
            frames=options.num_noise_levels * num_samples,
            interval=10,
        )
        plt.show()


if __name__ == "__main__":
    usage = "Usage: python double_integrator.py [generate|fit|deploy|gd]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)
    if sys.argv[1] == "gd":
        solve_with_gradient_descent()
    elif sys.argv[1] == "generate":
        generate_dataset()
    elif sys.argv[1] == "fit":
        fit_score_model()
    elif sys.argv[1] == "deploy":
        deploy_trained_model(plot=True, animate=True)
    else:
        print(usage)
        sys.exit(1)
