import numpy as np


def tournament(pop, fit, size, rng):
    picks = rng.choice(len(pop), size, replace=False)
    winner = picks[np.argmin(fit[picks])]
    return pop[winner]


def genetic_algorithm(problem, pop_size=20, iterations=40, crossover_rate=0.9,
                      mutation_rate=None, elite=2, tour_size=3, seed=None):
    rng = np.random.default_rng(seed)
    n = problem.n_features
    if mutation_rate is None:
        mutation_rate = 1.0 / n

    pop = rng.integers(0, 2, (pop_size, n))
    fit = np.array([problem.fitness(ind) for ind in pop])
    history = []

    for it in range(iterations):
        # elitism: the best individuals go to the next generation unchanged
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:elite]]

        while len(new_pop) < pop_size:
            p1 = tournament(pop, fit, tour_size, rng)
            p2 = tournament(pop, fit, tour_size, rng)

            # one point crossover
            if rng.random() < crossover_rate:
                point = rng.integers(1, n)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            # bit flip mutation
            for child in (c1, c2):
                flip = rng.random(n) < mutation_rate
                child[flip] = 1 - child[flip]
                new_pop.append(child)

        pop = np.array(new_pop[:pop_size])
        fit = np.array([problem.fitness(ind) for ind in pop])
        history.append(fit.min())

    best = np.argmin(fit)
    return pop[best], fit[best], history
