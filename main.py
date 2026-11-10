import argparse
import os
import time

# one thread for numpy, so the time of a run doesn't depend on how busy the other cores are
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset, DATASETS
from fitness import FeatureSelectionProblem, test_accuracy
from ga import genetic_algorithm
from pso import binary_pso
from aco import ant_colony
from baselines import all_features, forward_selection, exhaustive_search

ALGORITHMS = {
    "GA": genetic_algorithm,
    "PSO": binary_pso,
    "ACO": ant_colony,
}
COLORS = {"GA": "tab:blue", "PSO": "tab:orange", "ACO": "tab:green"}
OUT = "results"


def iterations_to_best(history):
    # first iteration where the final best value was reached (1-based)
    final = history[-1]
    for i, h in enumerate(history):
        if h <= final:
            return i + 1
    return len(history)


def evaluate(problem, data, name, algo, run, mask, fit, history, elapsed):
    X_train, X_test, y_train, y_test = data
    return {
        "dataset": name,
        "algorithm": algo,
        "run": run,
        "fitness": fit,
        "cv_accuracy": 1 - problem.error_rate(mask),
        "test_accuracy": test_accuracy(mask, X_train, y_train, X_test, y_test),
        "n_selected": int(mask.sum()),
        "time_s": elapsed,
        "evaluations": problem.evaluations,
        "iter_to_best": iterations_to_best(history) if history else None,
        "features": " ".join(str(i) for i in np.flatnonzero(mask)),
    }


def run_dataset(name, runs, pop, iters):
    data = load_dataset(name)
    X_train, X_test, y_train, y_test = data
    problem = FeatureSelectionProblem(X_train, y_train)
    rows, histories = [], {a: [] for a in ALGORITHMS}

    # the 3 algorithms run one after the other for each seed, so if the
    # computer gets slower for a while it affects all of them the same way
    for r in range(runs):
        for algo, func in ALGORITHMS.items():
            problem.cache = {}  # fresh cache so the time of every run is fair
            problem.evaluations = 0
            start = time.perf_counter()
            mask, fit, history = func(problem, pop, iters, seed=r)
            elapsed = time.perf_counter() - start
            histories[algo].append([float(h) for h in history])
            rows.append(evaluate(problem, data, name, algo, r, mask, fit, history, elapsed))
            print(f"  {algo} run {r + 1}/{runs}: fitness={fit:.4f} "
                  f"features={mask.sum()} time={elapsed:.1f}s")

    baselines = {"All features": all_features, "Forward selection": forward_selection}
    if problem.n_features <= 15:
        baselines["Exhaustive (optimal)"] = exhaustive_search

    for algo, func in baselines.items():
        problem.cache = {}
        problem.evaluations = 0
        start = time.perf_counter()
        mask, fit, history = func(problem)
        elapsed = time.perf_counter() - start
        rows.append(evaluate(problem, data, name, algo, 0, mask, fit, history, elapsed))
        print(f"  {algo}: fitness={fit:.4f} features={mask.sum()} time={elapsed:.1f}s")

    return rows, histories


def plot_convergence(name, histories, df):
    plt.figure(figsize=(7, 4.5))
    for algo, hist in histories.items():
        h = np.array(hist)
        mean = h.mean(axis=0)
        low, high = np.percentile(h, 25, axis=0), np.percentile(h, 75, axis=0)
        x = np.arange(1, len(mean) + 1)
        plt.plot(x, mean, label=algo, color=COLORS[algo])
        plt.fill_between(x, low, high, color=COLORS[algo], alpha=0.15)

    base = df[(df.dataset == name) & (df.algorithm.isin(["Forward selection", "Exhaustive (optimal)"]))]
    for _, row in base.iterrows():
        style = ":" if row.algorithm == "Forward selection" else "--"
        plt.axhline(row.fitness, color="gray", linestyle=style, label=row.algorithm)

    plt.title(f"Convergence - {DATASETS[name]}")
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (lower is better)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/convergence_{name}.png", dpi=120)
    plt.close()


def plot_boxplots(df, names):
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        d = df[df.dataset == name]
        ax.boxplot([d[d.algorithm == a].fitness for a in ALGORITHMS], tick_labels=list(ALGORITHMS))
        ax.set_title(DATASETS[name])
        ax.set_ylabel("Final best fitness")
        ax.grid(alpha=0.3)
    fig.suptitle("Stability over runs")
    fig.tight_layout()
    fig.savefig(f"{OUT}/boxplot_fitness.png", dpi=120)
    plt.close(fig)


def plot_bars(summary, names, column, ylabel, filename):
    algos = list(ALGORITHMS)
    x = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(7, 4.5))
    for i, algo in enumerate(algos):
        vals = [summary.loc[(n, algo), column] for n in names]
        plt.bar(x + (i - 1) * width, vals, width, label=algo, color=COLORS[algo])
    plt.xticks(x, [DATASETS[n] for n in names], fontsize=8)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}", dpi=120)
    plt.close()


def markdown_table(table):
    cols = ["algorithm"] + list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for algo, row in table.iterrows():
        lines.append("| " + " | ".join([algo] + [f"{v:g}" for v in row]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Feature selection with GA, PSO and ACO")
    parser.add_argument("--datasets", nargs="+", default=["small", "medium", "large"],
                        choices=list(DATASETS))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    all_rows, all_hist = [], {}
    for name in args.datasets:
        print(f"\n=== {DATASETS[name]} ===")
        rows, hist = run_dataset(name, args.runs, args.pop, args.iters)
        all_rows += rows
        all_hist[name] = hist

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{OUT}/runs.csv", index=False)

    summary = df.groupby(["dataset", "algorithm"], sort=False).agg(
        best_fitness=("fitness", "min"),
        mean_fitness=("fitness", "mean"),
        std_fitness=("fitness", "std"),
        worst_fitness=("fitness", "max"),
        mean_test_acc=("test_accuracy", "mean"),
        mean_features=("n_selected", "mean"),
        mean_time_s=("time_s", "mean"),
        mean_evaluations=("evaluations", "mean"),
        mean_iter_to_best=("iter_to_best", "mean"),
    ).fillna(0).round(4)
    summary.to_csv(f"{OUT}/summary.csv")

    with open(f"{OUT}/summary.md", "w") as f:
        for name in args.datasets:
            f.write(f"### {DATASETS[name]}\n\n")
            f.write(markdown_table(summary.loc[name]) + "\n\n")

    for name in args.datasets:
        plot_convergence(name, all_hist[name], df)
    plot_boxplots(df, args.datasets)
    plot_bars(summary, args.datasets, "mean_time_s", "Mean time per run (s)", "time_comparison.png")
    plot_bars(summary, args.datasets, "mean_test_acc", "Mean test accuracy", "test_accuracy.png")

    pd.set_option("display.width", 200)
    print("\n", summary[["best_fitness", "mean_fitness", "std_fitness", "mean_test_acc",
                        "mean_features", "mean_time_s", "mean_iter_to_best"]])
    print(f"\nSaved tables and plots to ./{OUT}/")


if __name__ == "__main__":
    main()
