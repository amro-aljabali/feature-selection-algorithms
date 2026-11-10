import numpy as np


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def binary_pso(problem, pop_size=20, iterations=40, w_max=0.9, w_min=0.4,
               c1=2.0, c2=2.0, v_max=4.0, seed=None):
    rng = np.random.default_rng(seed)
    n = problem.n_features

    x = rng.integers(0, 2, (pop_size, n))
    v = rng.uniform(-1, 1, (pop_size, n))
    fit = np.array([problem.fitness(p) for p in x])

    pbest = x.copy()
    pbest_fit = fit.copy()
    g = np.argmin(pbest_fit)
    gbest = pbest[g].copy()
    gbest_fit = pbest_fit[g]
    history = []

    for it in range(iterations):
        # inertia goes down linearly: explore first, then exploit
        w = w_max - (w_max - w_min) * it / iterations
        r1 = rng.random((pop_size, n))
        r2 = rng.random((pop_size, n))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -v_max, v_max)

        # sigmoid turns the velocity into the probability that the bit is 1
        x = (rng.random((pop_size, n)) < sigmoid(v)).astype(int)
        fit = np.array([problem.fitness(p) for p in x])

        better = fit < pbest_fit
        pbest[better] = x[better]
        pbest_fit[better] = fit[better]

        g = np.argmin(pbest_fit)
        if pbest_fit[g] < gbest_fit:
            gbest = pbest[g].copy()
            gbest_fit = pbest_fit[g]
        history.append(gbest_fit)

    return gbest, gbest_fit, history
