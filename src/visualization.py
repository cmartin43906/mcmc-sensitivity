import matplotlib.pyplot as plt

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
    plt.show()

def plot_sensitivity_analysis(t, A_sensitivity_norm, B_sensitivity_norm, A_sensitivity_t, B_sensitivity_t):
    print(f"Sensitivity norm for A: {A_sensitivity_norm:.4f}")
    print(f"Sensitivity norm for B: {B_sensitivity_norm:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(t, A_sensitivity_t, label="Sensitivity to A", linewidth=2)
    plt.plot(t, B_sensitivity_t, label="Sensitivity to B", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Finite-difference sensitivity")
    plt.title("Local Sensitivity of Jansen-Rit Output")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(["A", "B"], [A_sensitivity_norm, B_sensitivity_norm])
    plt.ylabel("Sensitivity norm")
    plt.title("Overall Parameter Sensitivity")
    plt.tight_layout()
    plt.show()

def plot_sensitivity_with_signal(t, y_clean, A_sensitivity_t, B_sensitivity_t):
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Left axis: sensitivity
    ax1.plot(t, A_sensitivity_t, label="Sensitivity to A", linewidth=2)
    ax1.plot(t, B_sensitivity_t, label="Sensitivity to B", linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Finite-difference sensitivity")
    ax1.set_title("Local Sensitivity with Jansen-Rit Output")

    # Right axis: original signal
    ax2 = ax1.twinx()
    ax2.plot(t, y_clean, color="black", linestyle="--", alpha=0.6, label="Clean signal")
    ax2.set_ylabel("EEG proxy")

    # combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.show()