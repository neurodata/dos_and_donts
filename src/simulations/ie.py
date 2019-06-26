import sys
import warnings
from itertools import product

import numpy as np
from tqdm import tqdm
from graspy.simulations import sample_edges
from scipy.stats import ttest_ind, wilcoxon

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

    def edge_significance(self, n_iter=10):
        """
        Calculate the significance of each edge

        Parameters
        ----------
        n_iter : int (default = 1)
            Number of Monte Carlo runs.

        Returns
        -------
        ttest_map : np.ndarray, shape (n, n)
            Array of spatially arranged p-values for T-test
        wilcoxon_map : np.ndarray, shape (n, n)
            Array of spatially arranged p-values for Wilcoxon
        """

        # Power matrices
        power_ttest = np.zeros(shape=(self.n_vertices, self.n_vertices))
        power_wilcoxon = np.zeros(shape=(self.n_vertices, self.n_vertices))

        for _ in tqdm(range(n_iter)):

            # Get samples
            x, y = self._sample()

            # Matrices to store p-values
            ttest_map = np.zeros(shape=(self.n_vertices, self.n_vertices))
            wilcoxon_map = np.zeros(shape=(self.n_vertices, self.n_vertices))

            for i, j in product(range(self.n_vertices), range(self.n_vertices)):

                xi = x[:, i, j]
                yi = y[:, i, j]

                if np.array_equal(xi, yi):
                    ttest_map[i, j] = wilcoxon_map[i, j] = 1
                else:
                    _, pval_1 = ttest_ind(xi, yi)
                    _, pval_2 = wilcoxon(xi, yi)
                    ttest_map[i, j] = pval_1
                    wilcoxon_map[i, j] = pval_2

            power_ttest += ttest_map < 0.05
            power_wilcoxon += wilcoxon_map < 0.05

        return power_ttest / n_iter, power_wilcoxon / n_iter
