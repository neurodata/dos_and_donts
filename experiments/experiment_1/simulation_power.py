# Vivek Gopalakrishnan
# July 8, 2019

# Test the effect of sample size on power.

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, fisher_exact

from src.simulations import IndependentEdge


def to_dataframe(ie, power, filename):

    df = pd.DataFrame(
        {
            r"$p_{ij}$": ie.p1.flatten(),
            "t-test": power[0].flatten(),
            "mann-whitney": power[1].flatten(),
            "fisher": power[2].flatten(),
            "boschloo": power[3].flatten(),
        }
    ).melt(
        id_vars=[r"$p_{ij}$"],
        value_vars=["t-test", "mann-whitney", "fisher", "boschloo"],
        var_name="test",
        value_name="power",
    )

    df.to_csv("results/power/{}.csv".format(filename))


if __name__ == "__main__":

    for sample_size in np.linspace(10, 100, 10):

        ie = IndependentEdge(
            sample_size=sample_size, n_vertices=10, epsilon=0.001, delta=0.05
        )
        pvals = ie.calculate_pvals(
            scipy_methods=[ttest_ind, mannwhitneyu, fisher_exact],
            r_methods=["boschloo",],
            n_iter=10,
        )
        power = ie.calculate_proportion_positive(pvals)

        filename = "m{}".format(int(sample_size))
        to_dataframe(ie, power, filename)
