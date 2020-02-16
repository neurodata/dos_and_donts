from argparse import ArgumentParser
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from graspy.plot import heatmap
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from mgc.ksample import KSample

from src import generate_binary_sbms, estimate_embeddings


def run_experiment(m, block_1, block_2, p, delta, n_components, reps):
    omni_corrects = np.zeros((reps, block_1 + block_2))
    mase_corrects = np.zeros((reps, block_1 + block_2))

    for i in np.arange(reps).astype(int):
        pop1, pop2, true_labels = generate_binary_sbms(
            m=m, block_1=block_1, block_2=block_2, p=p, delta=delta
        )

        for method in ["omni", "mase"]:
            embeddings = estimate_embeddings(
                pop1, pop2, method, n_components, sample_space=True
            )
            for j in range(block_1 + block_2):
                test_stat, pval = KSample("Dcorr").test(
                    embeddings[:m, j, :], embeddings[m:, j, :]
                )

                if method == "mase":
                    mase_corrects[i, j] = pval
                else:
                    omni_corrects[i, j] = pval

    omni_powers = (omni_corrects <= 0.05).mean(axis=0)
    mase_powers = (mase_corrects <= 0.05).mean(axis=0)

    to_append = [m, p, delta, *omni_powers, *mase_powers]
    return


def main(task_index):
    spacing = 50

    block_1 = 5  # different probability
    block_2 = 15
    p = 0.5
    deltas = np.linspace(0, 1 - p, spacing + 1)
    n_components = 2
    reps = 25
    ms = np.linspace(0, 500, spacing + 1)[1:].astype(int)

    partial_func = partial(
        run_experiment,
        block_1=block_1,
        block_2=block_2,
        p=p,
        reps=reps,
        n_components=n_components,
    )

    args = [dict(m=m, delta=delta) for m, delta in product(ms, deltas)]
    args = sum(zip(reversed(args), args), ())[: len(args)][task_index::10]

    res = Parallel(n_jobs=-1, verbose=5)(delayed(partial_func)(**arg) for arg in args)

    cols = [
        "m",
        "p",
        "delta",
        *[f"omni_power_node={i}" for i in range(block_1 + block_2)],
        *[f"mase_power_node={i}" for i in range(block_1 + block_2)],
    ]

    res_df = pd.DataFrame(res, columns=cols)
    res_df.to_csv(
        f"./results/20200216_weighted_correct_nodes_{task_index}.csv", index=False
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This is a script for running mgcx experiments."
    )
    parser.add_argument("task_index", help="SLURM task index")

    result = parser.parse_args()
    task_index = int(result.task_index)
    main(task_index)
