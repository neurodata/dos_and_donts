import warnings
from argparse import ArgumentParser
from itertools import product
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.plot import heatmap
from graspy.utils import symmetrize
from joblib import Parallel, delayed
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, multiscale_graphcorr

from src import generate_truncnorm_sbms, compute_pr_at_k

warnings.filterwarnings("ignore")


def compute_statistic(test, pop1, pop2):
    if test.__name__ == "ttest_ind":
        test_statistics, pvals = ttest_ind(pop1, pop2, axis=0)
        np.nan_to_num(test_statistics, copy=False)
        np.nan_to_num(pvals, copy=False)
    else:  # for other tests, do by edge
        n = pop1.shape[-1]
        test_statistics = np.zeros((n, n))
        pvals = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                x_ij = pop1[:, i, j]
                y_ij = pop2[:, i, j]

                if test.__name__ == "multiscale_graphcorr":
                    tmp, pval, _ = test(x=x_ij, y=y_ij, is_twosamp=True, reps=1)
                else:
                    tmp, pval = test(x=x_ij, y=y_ij)

                test_statistics[i, j] = tmp
                pvals[i, j] = pval

        test_statistics = symmetrize(test_statistics, method="triu")
        pvals = symmetrize(pvals, method="triu")

    return test_statistics, pvals


def run_experiment(tests, m, block_1, block_2, mean_1, mean_2, var_1, var_2, ks, reps):
    precisions = []
    recalls = []

    for _ in range(reps):
        tmp_precisions = []
        tmp_recalls = []
        pop1, pop2, true_labels = generate_truncnorm_sbms(
            m=m,
            block_1=block_1,
            block_2=block_2,
            mean_1=mean_1,
            mean_2=mean_2,
            var_1=var_1,
            var_2=var_2,
        )

        for test in tests:
            test_statistics, pvalues = compute_statistic(test, pop1, pop2)
            if test.__name__ == "multiscale_graphcorr":
                precision, recall = compute_pr_at_k(
                    k=ks, true_labels=true_labels, test_statistics=test_statistics
                )
            else:
                precision, recall = compute_pr_at_k(
                    k=ks, true_labels=true_labels, pvalues=pvalues
                )
            tmp_precisions.append(precision)
            tmp_recalls.append(recall)

        precisions.append(tmp_precisions)
        recalls.append(tmp_recalls)

    precisions = np.array(precisions).mean(axis=0)
    recalls = np.array(recalls).mean(axis=0)

    to_append = [m, mean_1, mean_2, var_1, var_2, *precisions, *recalls]

    return to_append


def main(task_index, num_arrays):
    tests = [ttest_ind, wilcoxon, mannwhitneyu, multiscale_graphcorr]

    spacing = 50
    block_1 = 5
    block_2 = 15
    mean_1 = 0
    mean_2 = 0
    var_1 = 1 / 2
    var_2s = np.linspace(var_1, 3, spacing + 1)
    ms = np.linspace(0, 500, spacing + 1).astype(int)[1:]
    ks = range(5, 11)
    reps = 100

    args = [dict(m=m, var_2=var_2) for (m, var_2) in product(ms, var_2s)][
        task_index::num_arrays
    ]

    partial_func = partial(
        run_experiment,
        tests=tests,
        block_1=block_1,
        block_2=block_2,
        mean_1=mean_1,
        mean_2=mean_2,
        var_1=var_1,
        ks=ks,
        reps=reps,
    )

    res = Parallel(n_jobs=-1, verbose=1)(delayed(partial_func)(**arg) for arg in args)

    cols = [
        "m",
        "mean1",
        "mean2",
        "var_1",
        "var_2",
        *[
            f"{test.__name__}_precision_at_{k}"
            for test in [multiscale_graphcorr]
            for k in ks
        ],
        *[
            f"{test.__name__}_recall_at_{k}"
            for test in [multiscale_graphcorr]
            for k in ks
        ],
    ]
    res_df = pd.DataFrame(res, columns=cols)
    res_df.to_csv(
        f"./results/20200213_changing_variances_results_{task_index}.csv", index=False
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This is a script for running mgcx experiments."
    )
    parser.add_argument("task_index", help="SLURM task index")
    parser.add_argument("num_arrays", help="SLURM Number of arrays")

    result = parser.parse_args()
    task_index = int(result.task_index)
    num_arrays = int(result.num_arrays)
    main(task_index, num_arrays)
