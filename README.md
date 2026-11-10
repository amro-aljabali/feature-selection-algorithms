# Feature Selection with GA, PSO and ACO

We pick the best subset of features for a KNN classifier using three
metaheuristics (Genetic Algorithm, Binary PSO, Ant Colony Optimization) and
compare them with simple baselines (all features, greedy forward selection,
and exhaustive search on the small dataset to get the real optimum).

## Setup

Python 3.9+.

```
pip install -r requirements.txt
```

## Run

Full experiment (3 datasets, 10 runs per algorithm, about 5-15 minutes depending on the laptop):

```
python main.py
```

Quick test (about 30 seconds):

```
python main.py --runs 2 --iters 10 --datasets small medium
```

Options:

| option | default | meaning |
|---|---|---|
| `--datasets` | `small medium large` | which datasets to use |
| `--runs` | 10 | independent runs per algorithm (seeds 0..runs-1) |
| `--pop` | 20 | population / swarm / colony size |
| `--iters` | 50 | number of iterations |

Results are reproducible: run `r` always uses seed `r`, and the dataset split is fixed.

## Output (`results/`)

- `runs.csv` - one line per run: fitness, CV accuracy, test accuracy, number of features, time, evaluations, selected features
- `summary.csv`, `summary.md` - mean / std / best / worst per algorithm and dataset
- `convergence_<dataset>.png` - iteration vs best fitness (line = mean over runs, shaded = 25%-75% of runs)
- `boxplot_fitness.png` - spread of the final fitness over the runs (stability)
- `time_comparison.png`, `test_accuracy.png` - bar charts

## Files

| file | content |
|---|---|
| `data.py` | loads the 3 datasets, 70/30 train/test split, scaling |
| `fitness.py` | the problem: binary solution, fitness function, KNN evaluation |
| `ga.py` | Genetic Algorithm |
| `pso.py` | Binary Particle Swarm Optimization |
| `aco.py` | Ant Colony Optimization |
| `baselines.py` | all features, forward selection, exhaustive search |
| `main.py` | runs everything, saves tables and plots |
| `report.md` | the report |

## Datasets

| name | dataset | samples | features | classes |
|---|---|---|---|---|
| small | Wine (sklearn) | 178 | 13 | 3 |
| medium | Breast Cancer Wisconsin (sklearn) | 569 | 30 | 2 |
| large | Synthetic, `make_classification` (15 informative, 10 redundant, 75 noise) | 600 | 100 | 2 |

All of them are built into scikit-learn, nothing to download.
