import itertools
import numpy as np


def all_features(problem):
    mask = np.ones(problem.n_features, dtype=int)
    return mask, problem.fitness(mask), []


def forward_selection(problem):
    # greedy: start empty, add the feature that helps the most, stop when nothing helps
    n = problem.n_features
    mask = np.zeros(n, dtype=int)
    best_fit = 1.0
    history = []

    while True:
        step_best, step_feature = best_fit, None
        for f in np.flatnonzero(mask == 0):
            mask[f] = 1
            fit = problem.fitness(mask)
            mask[f] = 0
            if fit < step_best:
                step_best, step_feature = fit, f
        if step_feature is None:
            break
        mask[step_feature] = 1
        best_fit = step_best
        history.append(best_fit)

    return mask, best_fit, history


def exhaustive_search(problem):
    # tries all 2^n - 1 subsets, only possible for small n (13 features -> 8191 subsets)
    n = problem.n_features
    best, best_fit = None, np.inf
    for bits in itertools.product([0, 1], repeat=n):
        mask = np.array(bits)
        fit = problem.fitness(mask)
        if fit < best_fit:
            best, best_fit = mask, fit
    return best, best_fit, []
