import sys
import warnings
from itertools import product

import numpy as np
from tqdm import tqdm
from graspy.simulations import sample_edges

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class IndependentEdge:
    def __init__(self, sample_size, n_vertices, epsilon, delta):

        self.sample_size = int(sample_size)
        self.n_vertices = int(n_vertices)
        self.epsilon = epsilon
        self.delta = delta

        self.p1, self.p2 = self._generate_p_matrices()

    def _generate_p_matrices(self):

        # Generate p1
        b = np.linspace(self.epsilon, 1 - self.epsilon, num=self.n_vertices ** 2)
        p1 = b.reshape(self.n_vertices, self.n_vertices)

        # Generate p2
        p2 = np.copy(p1)
        with np.nditer(p2, op_flags=["readwrite"]) as it:
            for x in it:
                if x < 0.5:
                    x[...] = x + self.delta
                else:
                    x[...] = x - self.delta

        return p1, p2

    def _sample(self):
        """
        Sample graphs from IE model

        Returns
        -------
        x, y : np.ndarray, shapes (m, n, n)
            Samples from two models
        """

        x = [
            sample_edges(self.p1, directed=True, loops=True)
            for _ in range(self.sample_size)
        ]
        y = [
            sample_edges(self.p2, directed=True, loops=True)
            for _ in range(self.sample_size)
        ]

        x = np.stack(x)
        y = np.stack(y)

        return x, y

    def calculate_power(self, *tests, n_iter=100):
        """
        Calculate the power of a given test

        Parameters
        ----------
        *tests : functions
            Statistical tests from scipy.stats
            Assumes function returns are of the form (statistic, p-value)
        n_iter : int (default = 1)
            Number of Monte Carlo runs.

        Returns
        -------
        power : np.ndarray, shape (n_tests, n_vertices, n_vertices)
            Proportion of tests that successfully rejected the null
        """

        # Power proportion matrix
        power = np.zeros(shape=(len(tests), self.n_vertices, self.n_vertices))

        for _ in tqdm(range(n_iter)):

            # Get samples
            x, y = self._sample()

            for idx, test in enumerate(tests):

                # Matrices to store p-values
                pvals = np.zeros(shape=(self.n_vertices, self.n_vertices))

                for i, j in product(range(self.n_vertices), range(self.n_vertices)):

                    xi = x[:, i, j]
                    yi = y[:, i, j]

                    if np.array_equal(xi, yi):
                        pval = 1
                    elif test.__name__ == "fisher_exact":
                        xi_n_zero = np.count_nonzero(xi)
                        yi_n_zero = np.count_nonzero(yi)
                        data = [
                            [xi_n_zero, self.sample_size - xi_n_zero],
                            [yi_n_zero, self.sample_size - yi_n_zero],
                        ]
                        _, pval = test(data)
                    else:
                        _, pval = test(xi, yi)

                    # Add pvalue to corresponding element in matrix
                    pvals[i, j] = pval

                # Tests with pvalues < 0.05 successfully rejected
                power[idx, :, :] += pvals < 0.05

        return power / n_iter
