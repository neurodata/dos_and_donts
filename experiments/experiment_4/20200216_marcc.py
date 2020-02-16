from argparse import ArgumentParser
from functools import partial
from itertools import product

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
from sklearn.metrics import adjusted_rand_score

from src import generate_truncnorm_sbms


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


def estimate_community(embeddings, true_labels, method, n_clusters):
    predicted_labels = GaussianCluster(n_clusters, n_clusters, "all").fit_predict(
        embeddings
    )

    # Label flipping
    idx = true_labels == 0
    if np.mean(predicted_labels[idx]) < 0.5:
        ari = adjusted_rand_score(true_labels, predicted_labels)
        return predicted_labels, ari
    else:
        # This is bitwise flipping. Turns all 0s to 1s and 1s to 0s.
        # Reason is to make labels consistent across repetitions
        # predicted_labels = predicted_labels ^ (predicted_labels & 1 == predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        return predicted_labels, ari


def run_experiment(m, block_1, block_2, mean_1, mean_2, var_1, var_2, n_clusters, reps):
    omni_corrects = np.zeros((reps, len(n_clusters) * 2))
    mase_corrects = np.zeros((reps, len(n_clusters) * 2))
    omni_aris = np.zeros((reps, len(n_clusters)))
    mase_aris = np.zeros((reps, len(n_clusters)))

    for i in np.arange(reps).astype(int):
        pop1, pop2, true_labels = generate_truncnorm_sbms(
            m=m,
            block_1=block_1,
            block_2=block_2,
            mean_1=mean_1,
            mean_2=mean_2,
            var_1=var_1,
            var_2=var_2,
        )
        mase_corrects_tmp = []
        omni_corrects_tmp = []
        mase_aris_tmp = []
        omni_aris_tmp = []
        for method in ["mase", "omni"]:
            embeddings = estimate_embeddings(pop1, pop2, method)

            for k in n_clusters:
                predicted_labels, ari = estimate_community(
                    embeddings, true_labels, method, k
                )

                uniques, counts = np.unique(
                    predicted_labels[:block_1], return_counts=True
                )
                b1_max_label = uniques[np.argmax(counts)]
                b1_correct = (predicted_labels[:block_1] == b1_max_label).mean()

                uniques, counts = np.unique(
                    predicted_labels[block_1:], return_counts=True
                )
                b2_max_label = uniques[np.argmax(counts)]
                b2_correct = (predicted_labels[block_1:] == b2_max_label).mean()

                if method == "mase":
                    mase_corrects_tmp += [b1_correct, b2_correct]
                    mase_aris_tmp += [ari]
                else:
                    omni_corrects_tmp += [b1_correct, b2_correct]
                    omni_aris_tmp += [ari]

        mase_corrects[i] = mase_corrects_tmp
        mase_aris[i] = mase_aris_tmp
        omni_corrects[i] = omni_corrects_tmp
        omni_aris[i] = omni_aris_tmp

    omni_powers = omni_corrects.mean(axis=0)
    omni_aris = omni_aris.mean(axis=0)
    mase_powers = mase_corrects.mean(axis=0)
    mase_aris = mase_aris.mean(axis=0)

    to_append = [
        m,
        mean_1,
        mean_2,
        var_1,
        var_2,
        *omni_powers,
        *omni_aris,
        *mase_powers,
        *mase_aris,
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
    var_2s = np.linspace(var_1, 3, spacing + 1)
    reps = 50
    n_clusters = range(2, 11, 2)
    ms = np.linspace(0, 250, spacing + 1)[1:].astype(int)

    partial_func = partial(
        run_experiment,
        block_1=block_1,
        block_2=block_2,
        mean_1=mean_1,
        mean_2=mean_2,
        var_1=var_1,
        n_clusters=n_clusters,
        reps=reps,
    )

    args = [dict(m=m, var_2=var_2) for m, var_2 in product(ms, var_2s)]
    args = sum(zip(reversed(args), args), ())[: len(args)]
    args = args[task_index::10]
    res = Parallel(n_jobs=-1, verbose=1)(delayed(partial_func)(**arg) for arg in args)

    cols = [
        "m",
        "mean_1",
        "mean_2",
        "var_1",
        "var_2",
        *[f"omni_correct_nodes_{i}_{k}" for k in n_clusters for i in range(1, 3)],
        *[f"omni_ari_{k}" for k in n_clusters],
        *[f"mase_correct_nodes_{i}_{k}" for k in n_clusters for i in range(1, 3)],
        *[f"mase_ari_{k}" for k in n_clusters],
    ]
    res_df = pd.DataFrame(res, columns=cols)
    res_df.to_csv(
        f"./results/20200216_weighted_correct_nodes_{task_index}.csv", index=False
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="This is a script for running experiments.")
    parser.add_argument("task_index", help="SLURM task index")

    result = parser.parse_args()
    task_index = result.task_index
    main(task_index)
