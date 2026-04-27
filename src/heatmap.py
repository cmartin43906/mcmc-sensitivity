import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mcmc import likelihood

def compute_likelihood_grid(observed_noisy, noise_std, t_end, sf):
    """
    Computes a grid of likelihood scores to inform the heatmap, using the likelihood function from MCMC.
    """
    A_vals = np.linspace(2.5, 3.4, 60)
    B_vals = np.linspace(18.0, 24.0, 60)

    LL = np.zeros((len(B_vals), len(A_vals)))

    # evaluate likelihood over grid
    for i, B in enumerate(B_vals):
        for j, A in enumerate(A_vals):
            params = {"A": A, "B": B}
            LL[i, j] = likelihood(params, observed_noisy, noise_std, t_end=t_end, sf=sf)

    return A_vals, B_vals, LL


def plot_likelihood_heatmap(A_vals, B_vals, LL):
    """
    Plots heatmap showing likelihood landscape.
    """
    plt.figure(figsize=(5, 5))

    im = plt.imshow(
        LL,
        origin="lower",
        aspect="auto",
        extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
    )

    plt.colorbar(im, label="Log-likelihood")

    plt.xlabel("A (Excitatory Gain)")
    plt.ylabel("B (Inhibitory Gain)")
    plt.title("Likelihood Surface")

    plt.tight_layout()
    sns.despine()
    plt.show()
