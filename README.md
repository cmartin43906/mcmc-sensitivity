# MCMC Sensitivity – Jansen–Rit

Sensitivity analysis and Bayesian parameter inference on a Jansen–Rit neural mass model to assess parameter identifiability (A: excitatory gain, B: inhibitory gain).

## Structure

mcmc-sensitivity/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── mcmc.py
│   ├── sensitivity_analysis.py
│   ├── heatmap.py
│   ├── visualization.py
│   ├── models/
│   │   └── jansenrit.py
│   └── data/
│       ├── time.npy
│       ├── y_clean.npy
│       └── y_observed.npy

## Setup

Create a virtual environment, then install dependencies:

python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt

## Run

From `src/`:

python3 main.py

## Methods

- Synthetic data: Jansen–Rit simulation + Gaussian noise
- Sensitivity Analysis: central finite differences, RMS norms
- Bayesian Inference: Metropolis MCMC, Gaussian likelihood with mean, flat prior

## Outputs

- Trace plots (A, B, log-likelihood)
- Posterior distributions
- Likelihood surface
- Sensitivity plot

## Notes

- Can alter parameters and tune MCMC in config.py
- MCMC plots may take a couple of minutes to generate.