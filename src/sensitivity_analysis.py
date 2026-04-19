import matplotlib.pyplot as plt
import numpy as np
from models.jansenrit import simulate_observation
from visualization import plot_sensitivity_analysis, plot_sensitivity_with_signal

def finite_difference_sensitivity(base_params, param_name, delta, t_end=4.0, sf=1000):
    """
    Runs finite difference sensitivity protocol to determine how sensitive model output is to a single parameter.

    Examines small push up vs. small push down, and generates overall sensitivity via root mean square.

    Returns time axis, sensitivity time trace, and scalar sensitivity.
    """
    params_plus = base_params.copy()
    params_minus = base_params.copy()

    # nudge the params up and down
    params_plus[param_name] += delta
    params_minus[param_name] -= delta

    # simulate with the adjusted params
    t, y_plus = simulate_observation(params=params_plus, t_end=t_end, sf=sf)
    _, y_minus = simulate_observation(params=params_minus, t_end=t_end, sf=sf)

    # approximate derivative of output with respect to parameter at each time point
    sensitivity_t = (y_plus - y_minus) / (2 * delta)

    # root mean square gives magnitidue of sensitivity over time series
    sensitivity_norm = np.sqrt(np.mean(sensitivity_t**2))

    return t, sensitivity_t, sensitivity_norm

def run_sensitivity_analysis(base_params, y_clean):
    """
    Runs sensitivity analysis for both parameters (gain of A and B) and plots results.
    """

    # how much to move each parameter
    delta_A = 0.1
    delta_B = 0.5

    # compute
    t, A_sensitivity_t, A_sensitivity_norm = finite_difference_sensitivity(
        base_params, "A", delta_A)
    _, B_sensitivity_t, B_sensitivity_norm = finite_difference_sensitivity(
        base_params, "B", delta_B)
    
    plot_sensitivity_analysis(t, A_sensitivity_norm, B_sensitivity_norm, A_sensitivity_t, B_sensitivity_t)

    plot_sensitivity_with_signal(t=t, y_clean=y_clean, A_sensitivity_t=A_sensitivity_t, B_sensitivity_t=B_sensitivity_t)

   # --- FFT comparison: clean signal vs A-sensitivity ---
   # to determine frequency content and period

    # # Time step from simulation output
    # dt = t[1] - t[0]

    # # Remove DC offset (mean) so the oscillatory peak is easier to see
    # y_clean_centered = y_clean - np.mean(y_clean)
    # A_sens_centered = A_sensitivity_t - np.mean(A_sensitivity_t)

    # # Frequency axes
    # freqs_clean = np.fft.rfftfreq(len(y_clean_centered), d=dt)
    # freqs_A = np.fft.rfftfreq(len(A_sens_centered), d=dt)

    # # Fourier transforms
    # fft_clean = np.fft.rfft(y_clean_centered)
    # fft_A = np.fft.rfft(A_sens_centered)

    # # Power spectra
    # power_clean = np.abs(fft_clean) ** 2
    # power_A = np.abs(fft_A) ** 2

    # # Ignore the zero-frequency bin when finding the dominant oscillation
    # idx_clean = np.argmax(power_clean[1:]) + 1
    # idx_A = np.argmax(power_A[1:]) + 1

    # print("Clean dominant frequency:", freqs_clean[idx_clean])
    # print("A sensitivity dominant frequency:", freqs_A[idx_A])

    # print("Clean dominant period:", 1 / freqs_clean[idx_clean])
    # print("A sensitivity dominant period:", 1 / freqs_A[idx_A])

    # # Optional: plot both spectra for visual comparison

    # plt.figure(figsize=(10, 4))
    # plt.plot(freqs_clean, power_clean, label="Clean signal", linewidth=2)
    # plt.plot(freqs_A, power_A, label="A sensitivity", linewidth=2)
    # plt.xlim(0, 10)  # adjust if needed
    # plt.xlabel("Frequency")
    # plt.ylabel("Power")
    # plt.title("FFT Comparison: Clean Signal vs A Sensitivity")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
