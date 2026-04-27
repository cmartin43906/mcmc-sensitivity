import numpy as np
from models.jansenrit import simulate_observation
from visualization import plot_sensitivity_with_signal
from config import T_END, SF


def finite_difference_sensitivity(base_params, param_name, delta, t_end=T_END, sf=SF):
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
        base_params, "A", delta_A
    )
    _, B_sensitivity_t, B_sensitivity_norm = finite_difference_sensitivity(
        base_params, "B", delta_B
    )

    plot_sensitivity_with_signal(
        t=t,
        y_clean=y_clean,
        A_sensitivity_t=A_sensitivity_t,
        B_sensitivity_t=B_sensitivity_t,
    )

    print(f"A Sensitivity norm: {A_sensitivity_norm}\n")
    print(f"B Sensitivity Norm: {B_sensitivity_norm}\n")
