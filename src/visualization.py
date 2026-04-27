import matplotlib.pyplot as plt
import seaborn as sns

from config import burn
from models.jansenrit import simulate_observation


def plot_base_noisy(t, y_clean, observed_noisy):
    plt.figure(figsize=(8, 4))
    plt.plot(t, y_clean, label="Clean signal", linewidth=2)
    plt.plot(t, observed_noisy, label="Noisy observation", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("EEG proxy")
    plt.title("Synthetic Jansen-Rit Data")
    plt.legend()
    plt.tight_layout()
    sns.despine()
    plt.show()


def qual_param_sweep(param_sets):
    plt.figure(figsize=(10, 5))

    for label, params in param_sets.items():
        t, y = simulate_observation(params=params)
        plt.plot(t, y, label=label, linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("EEG proxy")
    plt.title("Effect of A and B on Jansen-Rit Output")
    plt.legend()
    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_sensitivity_with_signal(t, y_clean, A_sensitivity_t, B_sensitivity_t):
    _, ax1 = plt.subplots(figsize=(10, 4))

    # left axis: sensitivity
    ax1.plot(t, A_sensitivity_t, label="Sensitivity to A", linewidth=2)
    ax1.plot(t, B_sensitivity_t, label="Sensitivity to B", linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Finite-difference sensitivity")
    ax1.set_title("Local Sensitivity with Jansen-Rit Output")

    # right axis: original signal
    ax2 = ax1.twinx()
    ax2.plot(t, y_clean, color="black", linestyle="--", alpha=0.6, label="Clean signal")
    ax2.set_ylabel("EEG proxy")

    # combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_mcmc_traces(A_chain, B_chain, logL_chain):
    # trace plots first
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    fig.suptitle("Trace Plots of MCMC for Jansen-Rit Parameter Inference", fontsize=14)

    axes[0].plot(A_chain, linewidth=1)
    axes[0].axhline(3.25, color="red", linestyle="--", label="True A")
    axes[0].set_ylabel("A")
    axes[0].set_title("Trace of Excitatory Gain (A)")
    axes[0].legend()

    axes[1].plot(B_chain, linewidth=1)
    axes[1].axhline(22.0, color="red", linestyle="--", label="True B")
    axes[1].set_ylabel("B")
    axes[1].set_title("Trace of Inhibitory Gain (B)")
    axes[1].legend()

    axes[2].plot(logL_chain, linewidth=1)
    axes[2].set_ylabel("Log-Likelihood")
    axes[2].set_title("Trace of Log-Likelihood")
    axes[2].set_xlabel("MCMC Step")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    sns.despine()
    plt.show()

    # set burn-in
    A_post = A_chain[burn:]
    B_post = B_chain[burn:]

    plt.figure(figsize=(5, 5))
    plt.scatter(A_post, B_post, s=6, alpha=0.3)
    plt.axvline(3.25, color="red", linestyle="--")
    plt.axhline(22.0, color="red", linestyle="--")
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title("Joint Posterior Samples of A and B After Burn-in")
    plt.tight_layout()
    sns.despine()
    plt.show()
