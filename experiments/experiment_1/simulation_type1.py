# Vivek Gopalakrishnan
# July 9, 2019

# Test the effect of sample size on type 1 error.

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, fisher_exact

from src.simulations import IndependentEdge


def to_dataframe(ie, type1, filename):

    df = pd.DataFrame(
        {
            r"$p_{ij}$": ie.p1.flatten(),
            "t-test": type1[0].flatten(),
            "mann-whitney": type1[1].flatten(),
            "fisher": type1[2].flatten(),
            "boschloo": type1[3].flatten(),
        }
    ).melt(
        id_vars=[r"$p_{ij}$"],
        value_vars=["t-test", "mann-whitney", "fisher", "boschloo"],
        var_name="test",
        value_name="type1",
    )

    df.to_csv("results/type1/{}.csv".format(filename))


if __name__ == "__main__":

    for sample_size in np.linspace(10, 100, 10):

        ie = IndependentEdge(sample_size=sample_size, n_vertices=10, epsilon=0, delta=0)
        pvals = ie.calculate_pvals(
            scipy_methods=[ttest_ind, mannwhitneyu, fisher_exact],
            r_methods=["boschloo"],
            n_iter=10,
        )
        type1 = ie.calculate_proportion_positive(pvals)

        filename = "m{}".format(int(sample_size))
        to_dataframe(ie, type1, filename)
