from models.jansenrit import simulate_observation

import numpy as np

# likelihood needs guess params, noisy data, and noise level sigma

def likelihood(params, observed_noisy, sigma, t_end=4.0, sf=1000):

    # simulate using guesses
    t, y_sim = simulate_observation(params=params, t_end=t_end, sf=sf)

    # between observed and guess data
    error = observed_noisy - y_sim

    # Gaussian log_likelihood
    log_likelihood = -np.mean(error**2) / (2 * sigma**2)

    return log_likelihood

def run_mcmc(observed_noisy, sigma, num_steps=5000, init_params=None, step_sizes=None):
    """
    observed_noisy = observed data we are inferring from
    sigma = noise std
    num_steps = how long to run the markov chain
    init_params = where the markov chain starts
    step_size = how big each random step should be
    """

    if init_params is None:
        init_params = {"A": 3.2, "B":21.5}
    if step_sizes is None:
        step_sizes = {"A": 0.007, "B": 0.03}

    # storage arrays
    A_chain = np.zeros(num_steps)
    B_chain = np.zeros(num_steps)
    logL_chain = np.zeros(num_steps)

    # find likelihood of initial params
    curr_params = init_params.copy()
    current_logL = likelihood(curr_params, observed_noisy, sigma)

    # set index 0 of the chains
    A_chain[0] = curr_params["A"]
    B_chain[0] = curr_params["B"]
    logL_chain[0] = current_logL

    n_accept = 0
    for i in range(1, num_steps):
        # generate the candidate values
        candidates = curr_params.copy()
        candidates["A"] += np.random.normal(0, step_sizes["A"])
        candidates["B"] += np.random.normal(0, step_sizes["B"])

        # if the produced candidate values are invalid, keep the previous
        if candidates["A"] <= 0 or candidates["B"] <= 0:
            A_chain[i] = curr_params["A"]
            B_chain[i] = curr_params["B"]
            logL_chain[i] = current_logL
            continue

        candidate_logL = likelihood(candidates, observed_noisy, sigma)

        alpha = candidate_logL - current_logL

        if alpha >= 0:
            curr_params = candidates
            current_logL = candidate_logL
            n_accept += 1
        else:
            if np.log(np.random.rand()) < alpha:
                curr_params = candidates
                current_logL = candidate_logL
                n_accept += 1

        A_chain[i] = curr_params["A"]
        B_chain[i] = curr_params["B"]
        logL_chain[i] = current_logL


    acceptance_rate = n_accept / (num_steps - 1)
    return A_chain, B_chain, logL_chain, acceptance_rate