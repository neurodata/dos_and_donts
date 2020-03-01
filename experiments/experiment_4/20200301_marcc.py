from argparse import ArgumentParser
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from graspy.cluster import GaussianCluster
from graspy.embed import MultipleASE, OmnibusEmbed
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, mannwhitneyu, multiscale_graphcorr, ttest_ind

from src import generate_truncnorm_sbms_with_communities


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

    # ari = adjusted_rand_score(true_labels, predicted_labels)
    return predicted_labels


def compute_statistic(tests, pop1, pop2):
    res = np.zeros(len(tests))

    for idx, test in enumerate(tests):
        if test.__name__ == "multiscale_graphcorr":
            statistic, pval, _ = test(pop1, pop2, reps=250, is_twosamp=True)
        else:  # for other tests, do by edge
            statistic, pval = test(pop1, pop2)
        res[idx] = pval

    return res


def run_experiment(
    m,
    block_1,
    block_2,
    mean_1,
    mean_2,
    var_1,
    var_2,
    mean_delta,
    var_delta,
    n_clusters,
    reps,
    tests,
):
    total_n = block_1 + block_2
    r, c = np.triu_indices(total_n, k=1)

    omni_res = np.zeros((reps, len(n_clusters), 2, len(tests)))
    mase_res = np.zeros((reps, len(n_clusters), 2, len(tests)))

    for i in np.arange(reps).astype(int):
        pop1, pop2, true_labels = generate_truncnorm_sbms_with_communities(
            m=m,
            block_1=block_1,
            block_2=block_2,
            mean_1=mean_1,
            mean_2=mean_2,
            var_1=var_1,
            var_2=var_2,
            mean_delta=mean_delta,
            var_delta=var_delta,
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
                sig_edges = np.zeros((len(tests), total_n, total_n))[:, r, c]

                for cluster_label in np.unique(predicted_edge_labels):
                    tmp_labels = predicted_edge_labels == cluster_label
                    tmp_pop1_edges = pop1_edges[:, tmp_labels].ravel()
                    tmp_pop2_edges = pop2_edges[:, tmp_labels].ravel()

                    pvals = compute_statistic(tests, tmp_pop1_edges, tmp_pop2_edges)
                    for p_idx, pval in enumerate(pvals):
                        if pval <= 0.05:
                            sig_edges[p_idx][tmp_labels] = 1

                prec = (sig_edges[:, true_edges == 0]).sum(axis=1) / sig_edges.sum(
                    axis=1
                )
                np.nan_to_num(prec, False)
                recall = (sig_edges[:, true_edges == 0]).sum(axis=1) / (
                    true_edges == 0
                ).sum(axis=0)

                if method == "mase":
                    mase_res[i, k_idx, :] = np.array((prec, recall))
                else:
                    omni_res[i, k_idx, :] = np.array((prec, recall))

    omni_res = omni_res.mean(axis=0).reshape(-1)
    mase_res = mase_res.mean(axis=0).reshape(-1)

    to_append = [
        m,
        mean_1,
        mean_2,
        var_1,
        var_2,
        mean_delta,
        var_delta,
        *omni_res,
        *mase_res,
    ]
    return to_append


def main(task_index):
    task_index = int(task_index)

    spacing = 50

    block_1 = 25  # different probability
    block_2 = 25
    mean_1 = 0
    mean_2 = 0
    var_1 = 1 / 2
    var_2 = 1 / 2
    mean_delta = 0
    var_deltas = np.linspace(var_1, 3, spacing + 1)
    reps = 100
    n_clusters = range(2, 11)
    ms = np.linspace(0, 250, spacing + 1)[1:].astype(int)

    tests = [ks_2samp, mannwhitneyu, multiscale_graphcorr, ttest_ind]

    partial_func = partial(
        run_experiment,
        block_1=block_1,
        block_2=block_2,
        mean_1=mean_1,
        mean_2=mean_2,
        var_1=var_1,
        var_2=var_2,
        mean_delta=mean_delta,
        n_clusters=n_clusters,
        reps=reps,
        tests=tests,
    )

    args = [dict(m=m, var_delta=var_delta) for m, var_delta in product(ms, var_deltas)]
    args = sum(zip(reversed(args), args), ())[: len(args)]
    args = args[task_index::10]
    res = Parallel(n_jobs=-1, verbose=7)(delayed(partial_func)(**arg) for arg in args)

    cols = [
        "m",
        "mean_1",
        "mean_2",
        "var_1",
        "var_2",
        "mean_delta",
        "var_delta",
        *[
            f"omni_{metric}_{k}_{test.__name__}"
            for k in n_clusters
            for metric in ["precision", "recall"]
            for test in tests
        ],
        *[
            f"mase_{metric}_{k}_{test.__name__}"
            for k in n_clusters
            for metric in ["precision", "recall"]
            for test in tests
        ],
    ]
    res_df = pd.DataFrame(res, columns=cols)
    res_df.to_csv(
        f"./results/20200301_weighted_correct_nodes_{task_index}.csv", index=False
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="This is a script for running experiments.")
    parser.add_argument("task_index", help="SLURM task index")

    result = parser.parse_args()
    task_index = result.task_index
    main(task_index)
