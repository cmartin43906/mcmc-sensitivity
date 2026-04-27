T_END = 3.5
SF = 1000
num_steps = 8000
burn = 2000

true_params = {
    "A": 3.25,
    "B": 22.0,
}

param_sets = {
    "baseline": {"A": 3.25, "B": 22.0},
    "A up": {"A": 3.50, "B": 22.0},
    "A down": {"A": 3.00, "B": 22.0},
    "B up": {"A": 3.25, "B": 24.0},
    "B down": {"A": 3.25, "B": 20.0},
}

init_params = {"A": 3.15, "B": 23.5}

step_sizes = {"A": 0.02, "B": 0.08}  # 0.02  # 0.08

# 0.01 and 0.04 with 1.5s produce 16% rate, but weirder trace plots
