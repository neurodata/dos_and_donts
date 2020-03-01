from argparse import ArgumentParser
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from graspy.cluster import GaussianCluster
from graspy.embed import MultipleASE, OmnibusEmbed
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from src import generate_binary_sbms_with_communities


def estimate_embeddings(X, Y, method):
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

    return embeddings


def estimate_community(embeddings, n_clusters):
    predicted_labels = (
        GaussianCluster(n_clusters, n_clusters, "all").fit_predict(embeddings) + 1
    )
    return predicted_labels


def compute_ttest(pop1, pop2):
    statistics, pvals = ttest_ind(pop1, pop2, axis=0)
    return pvals


def run_experiment(m, block_1, block_2, p, q, delta, n_clusters, reps):
    total_n = block_1 + block_2
    r, c = np.triu_indices(total_n, k=1)

    omni_res = np.zeros((reps, len(n_clusters), 2))
    mase_res = np.zeros((reps, len(n_clusters), 2))

    for i in np.arange(reps).astype(int):
        pop1, pop2, true_labels = generate_binary_sbms_with_communities(
            m=m, block_1=block_1, block_2=block_2, p=p, q=q, delta=delta
        )
        pop1_edges = pop1[:, r, c]
        pop2_edges = pop2[:, r, c]
        true_edges = (true_labels[:, None] + true_labels[None, :])[r, c]

        for method in ["mase", "omni"]:
            embeddings = estimate_embeddings(pop1, pop2, method)

            for k_idx, k in enumerate(n_clusters):
                predicted_labels = estimate_community(embeddings, k)
                predicted_edge_labels = (
                    predicted_labels[:, None] * predicted_labels[None, :]
                )[
                    r, c
                ]  # vectorize to uppper triu
                sig_edges = np.zeros((total_n, total_n))[r, c]

                for cluster_label in np.unique(predicted_edge_labels):
                    tmp_labels = predicted_edge_labels == cluster_label
                    statistics, pvals = ttest_ind(
                        pop1_edges[:, tmp_labels].ravel(),
                        pop2_edges[:, tmp_labels].ravel(),
                    )
                    if pvals <= 0.05:
                        sig_edges[tmp_labels] = 1

                prec = (sig_edges[true_edges == 0]).sum() / sig_edges.sum()
                recall = (sig_edges[true_edges == 0]).sum() / (true_edges == 0).sum()

                if method == "mase":
                    mase_res[i, k_idx, :] = (prec, recall)
                else:
                    omni_res[i, k_idx, :] = (prec, recall)

    omni_res = omni_res.mean(axis=0).reshape(-1)
    mase_res = mase_res.mean(axis=0).reshape(-1)

    to_append = [m, p, q, delta, *omni_res, *mase_res]
    return to_append


def main(task_index):
    task_index = int(task_index)

    block_1 = 25  # different probability
    block_2 = 25
    n_clusters = range(2, 11)
    p = 0.5
    q = 0.25
    reps = 100
    spacing = 50
    deltas = np.linspace(0, 1 - p, spacing + 1)
    ms = np.linspace(0, 250, spacing + 1)[1:]

    partial_func = partial(
        run_experiment,
        block_1=block_1,
        block_2=block_2,
        p=p,
        q=q,
        n_clusters=n_clusters,
        reps=reps,
    )
    args = [dict(m=m, delta=delta) for m, delta in product(ms, deltas)][task_index::10]
    args = sum(zip(reversed(args), args), ())[
        : len(args)
    ]  # This is to prevent memory error

    res = Parallel(n_jobs=-1, verbose=7)(delayed(partial_func)(**arg) for arg in args)

    cols = [
        "m",
        "p",
        "q",
        "delta",
        *[
            f"omni_{metric}_{k}"
            for k in n_clusters
            for metric in ["precision", "recall"]
        ],
        *[
            f"mase_{metric}_{k}"
            for k in n_clusters
            for metric in ["precision", "recall"]
        ],
    ]
    res_df = pd.DataFrame(res, columns=cols)

    res_df.to_csv(f"./results/20200301_correct_nodes_{task_index}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="This is a script for running experiments.")
    parser.add_argument("task_index", help="SLURM task index")

    result = parser.parse_args()
    task_index = result.task_index
    main(task_index)
