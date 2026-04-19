import numpy as np
import matplotlib.pyplot as plt
from models.jansenrit import solve_jr
from visualization import *
from sensitivity_analysis import *

true_params = {
    "A": 3.25,   # excitatory gain
    "B": 22.0,   # inhibitory gain
}

param_sets = {
    "baseline": {"A": 3.25, "B": 22.0},
    "A up": {"A": 3.50, "B": 22.0},
    "A down": {"A": 3.00, "B": 22.0},
    "B up": {"A": 3.25, "B": 24.0},
    "B down": {"A": 3.25, "B": 20.0},
}



def main():
    t, y_clean = simulate_observation(params=true_params)

    np.random.seed(27)
    noise_std = 0.15 * np.std(y_clean)
    noise = np.random.normal(0, noise_std, size=y_clean.shape)

    observed_noisy = y_clean + noise

    plot_base_noisy(t=t, y_clean=y_clean, observed_noisy=observed_noisy)

    np.save("data/time.npy", t)
    np.save("data/y_clean.npy", y_clean)
    np.save("data/y_observed.npy", observed_noisy)

    qual_param_sweep(param_sets=param_sets)

    run_sensitivity_analysis(base_params=true_params, y_clean=y_clean)


if __name__ == "__main__":
    main()