import numpy as np
from sklearn.feature_selection import mutual_info_classif


def ant_colony(problem, n_ants=20, iterations=40, alpha=1.0, beta=1.0, rho=0.2,
               tau_min=0.05, seed=None):
    """
    Each feature has two pheromone values: one for "take it" and one for
    "leave it". An ant walks over the features and decides for each one.
    Heuristic info (eta) = mutual information between the feature and the class.
    """
    rng = np.random.default_rng(seed)
    n = problem.n_features

    mi = mutual_info_classif(problem.X, problem.y, random_state=0)
    eta = 0.1 + 0.8 * (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)  # in [0.1, 0.9]

    tau_on = np.ones(n)
    tau_off = np.ones(n)

    best = None
    best_fit = np.inf
    history = []

    for it in range(iterations):
        a = (tau_on ** alpha) * (eta ** beta)
        b = (tau_off ** alpha) * ((1 - eta) ** beta)
        p_take = a / (a + b)

        ants = (rng.random((n_ants, n)) < p_take).astype(int)
        fit = np.array([problem.fitness(ant) for ant in ants])

        i = np.argmin(fit)
        if fit[i] < best_fit:
            best = ants[i].copy()
            best_fit = fit[i]

        # evaporation + deposit on the decisions of the best ant so far
        quality = 1 - best_fit
        tau_on = (1 - rho) * tau_on + rho * quality * (best == 1)
        tau_off = (1 - rho) * tau_off + rho * quality * (best == 0)
        tau_on = np.clip(tau_on, tau_min, 1.0)
        tau_off = np.clip(tau_off, tau_min, 1.0)

        history.append(best_fit)

    return best, best_fit, history
