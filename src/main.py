import numpy as np
from visualization import *
from sensitivity_analysis import *
from mcmc import run_mcmc
from config import *
from heatmap import *


def main():
    print("Bayesian Inference and Sensitivity Analysis Demo.\n\n")

    # generation of synthetic data
    t, y_clean = simulate_observation(params=true_params)
    np.random.seed(27)
    noise_std = 0.15 * np.std(y_clean)
    noise = np.random.normal(0, noise_std, size=y_clean.shape)
    observed_noisy = y_clean + noise

    print("Plot of Synthetic Data:\n\n")
    plot_base_noisy(t=t, y_clean=y_clean, observed_noisy=observed_noisy)

    # save the data
    np.save("data/time.npy", t)
    np.save("data/y_clean.npy", y_clean)
    np.save("data/y_observed.npy", observed_noisy)

    print("Qualitative Parameter Sweep:\n\n")
    qual_param_sweep(param_sets=param_sets)

    print("Finite Central Difference Sensitivity Analysis:\n\n")
    run_sensitivity_analysis(base_params=true_params, y_clean=y_clean)

    # run likelihood scan to generate heatmap
    print("Running Log-Likelihood scan. May take a moment...\n\n")
    A_vals, B_vals, LL = compute_likelihood_grid(
        observed_noisy, noise_std, t_end=T_END, sf=1000
    )

    plot_likelihood_heatmap(A_vals, B_vals, LL)

    # run MCMC to generate variables to plot, with acceptance rate
    print("Running MCMC. May take a moment...\n\n")
    A_chain, B_chain, logL_chain, accept_rate = run_mcmc(
        init_params=init_params,
        observed_noisy=observed_noisy,
        sigma=noise_std,
        step_sizes=step_sizes,
        num_steps=num_steps,
    )

    print("Acceptance rate:", accept_rate)

    plot_mcmc_traces(A_chain, B_chain, logL_chain)

    print("\n\nDemo complete.")


if __name__ == "__main__":
    main()
