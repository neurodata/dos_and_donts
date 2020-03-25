from argparse import ArgumentParser
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from hyppo.ksample import Hotelling, KSample

from src import generate_binary_sbms


def run_experiment(m, block_1, block_2, p, delta, reps, tests):
    pvals = np.zeros((reps, block_1 + block_2, len(tests)))

    for i in range(reps):
        X, Y, _ = generate_binary_sbms(m, block_1, block_2, p, delta)
        for j in range(block_1 + block_2):
            for k, test in enumerate(tests):
                X_nodes = np.delete(X[:, j, :], j, axis=1)
                Y_nodes = np.delete(Y[:, j, :], j, axis=1)
                try:
                    stat, pval = test.test(X_nodes, Y_nodes, reps=100)
                except:
                    pval = 1
                pvals[i, j, k] = pval

    pvals = (pvals <= 0.05).mean(axis=0).reshape(-1)
    to_append = [m, p, delta, *pvals]
    return to_append


def main(task_index):
    # Experiment Parameters
    # Constants
    block_1 = 5
    block_2 = 15
    p = 0.5
    reps = 25
    tests = [KSample("Dcorr"), Hotelling()]

    # Varying
    spacing = 50
    deltas = np.linspace(0, 1 - p, spacing + 1)
    ms = np.linspace(0, 500, spacing + 1)[1:]

    args = [dict(m=m, delta=delta) for m, delta in product(ms, deltas)]
    # Task subsample
    args = args[task_index::10]
    # args = sum(zip(reversed(args), args), ())[: len(args)]

    partial_func = partial(
        run_experiment, block_1=block_1, block_2=block_2, p=p, reps=reps, tests=tests
    )

    res = Parallel(n_jobs=-2, verbose=7)(delayed(partial_func)(**arg) for arg in args)

    cols = cols = ["m", "p", "delta", *[f"node_{i}" for i in range(1, 21)]]
    res_df = pd.DataFrame(res, columns=cols)
    res_df.to_csv(f"./results/20200305_adj_row_{task_index}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This is a script for running mgcx experiments."
    )
    parser.add_argument("task_index", help="SLURM task index")

    result = parser.parse_args()
    task_index = int(result.task_index)
    main(task_index)
