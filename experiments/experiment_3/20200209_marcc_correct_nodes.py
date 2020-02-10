from argparse import ArgumentParser
from functools import partial
from itertools import product


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.cluster import GaussianCluster
from graspy.embed import MultipleASE, OmnibusEmbed
from graspy.models import SBMEstimator
from graspy.plot import heatmap
from graspy.simulations import er_np, sbm
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from src import generate_binary_sbms


def estimate_community(X, Y, true_labels, method, n_clusters):
    """
    Parameters
    ----------
    method : str
        Must be {'mase', 'omni'}
    """
    stacked = np.vstack([X, Y])

    if method == "mase":
        embedder = MultipleASE(2)
        embeddings = embedder.fit_transform(stacked)
    elif method == "omni":
        embedder = OmnibusEmbed(2)
        embeddings = embedder.fit_transform(stacked).mean(axis=0)
    else:
        assert ValueError("Invalid embedding method")

    predicted_labels = GaussianCluster(n_clusters, n_clusters, "all").fit_predict(
        embeddings
    )

    # Label flipping
    idx = true_labels == 0
    if np.mean(predicted_labels[idx]) < 0.5:
        return predicted_labels
    else:
        # This is bitwise flipping. Turns all 0s to 1s and 1s to 0s.
        # Reason is to make labels consistent across repetitions
        predicted_labels = predicted_labels ^ (predicted_labels & 1 == predicted_labels)
        return predicted_labels


def estimate_b(X, Y, labels):
    pop_1_block_p = np.array([SBMEstimator(False).fit(x, labels).block_p_ for x in X])
    pop_2_block_p = np.array([SBMEstimator(False).fit(x, labels).block_p_ for x in Y])

    return pop_1_block_p, pop_2_block_p


def compute_ttest(pop1, pop2):
    statistics, pvals = ttest_ind(pop1, pop2, axis=0)

    return pvals


def run_experiment(m, block_1, block_2, p, delta, n_clusters, reps):
    omni_powers = np.zeros((reps, 2))
    mase_powers = np.zeros((reps, 2))

    for i in np.arange(reps).astype(int):
        pop1, pop2, true_labels = generate_binary_sbms(
            m=m, block_1=block_1, block_2=block_2, p=p, delta=delta
        )
        for method in ["mase", "omni"]:
            predicted_labels = estimate_community(
                pop1, pop2, true_labels, method, n_clusters
            )
            # pop1_b, pop2_b = estimate_b(pop1, pop2, predicted_labels)
            # pvals = compute_ttest(pop1_b, pop2_b)
            b1_correct = (predicted_labels[:block_1] == true_labels[:block_1]).mean()
            b2_correct = (predicted_labels[block_1:] == true_labels[block_1:]).mean()

            if method == "mase":
                mase_powers[i] = [b1_correct, b2_correct]
            else:
                omni_powers[i] = [b1_correct, b2_correct]
    # omni_powers = (omni_powers <= 0.05).mean(axis=0)[np.triu_indices(2)]
    # mase_powers = (mase_powers <= 0.05).mean(axis=0)[np.triu_indices(2)]

    to_append = [m, p, delta, *omni_powers, *mase_powers]
    return to_append


def main(task_index):
    block_1 = 25  # different probability
    block_2 = 25
    n_clusters = 2
    p = 0.5
    reps = 100

    spacing = 50
    deltas = np.linspace(0, 1 - p, spacing + 1)
    ms = np.linspace(0, 500, spacing + 1)[1:]

    partial_func = partial(
        run_experiment,
        block_1=block_1,
        block_2=block_2,
        p=p,
        n_clusters=n_clusters,
        reps=reps,
    )
    args = [dict(m=m, delta=delta) for m, delta in product(ms, deltas)][task_index::10]

    res = Parallel(n_jobs=-1, verbose=1)(delayed(partial_func)(**arg) for arg in args)

    cols = [
        "m",
        "p",
        "delta",
        *[f"omni_correct_nodes_{i}" for i in range(1, 3)],
        *[f"mase_correct_nodes_{i}" for i in range(1, 3)],
    ]
    res_df = pd.DataFrame(res, columns=cols)

    res_df.to_csv(f"./results/20200209_correct_nodes_{task_index}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This is a script for running mgcx experiments."
    )
    parser.add_argument("task_index", help="SLURM task index")

    result = parser.parse_args()
    task_index = result.task_index
    main(task_index)
