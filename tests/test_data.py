import numpy as np
import pytest
from joblib import Parallel, delayed

from src import generate_binary_sbms, generate_truncnorm_sbms


def test_binary_sbms():
    m = 10
    block_1 = 5
    block_2 = 15
    p = 0.5
    delta = 0.25

    res = Parallel(n_jobs=2, backend="loky")(
        delayed(generate_binary_sbms)(m, block_1, block_2, p, delta) for _ in range(2)
    )

    X1, Y1 = res[0]
    X2, Y2 = res[1]

    for i in range(m):
        assert ~np.all(X1[i] == X2[i])
        assert ~np.all(Y1[i] == Y2[i])


def test_truncnorm_sbms():
    m = 10
    block_1 = 5
    block_2 = 15
    mean_1 = 0
    mean_2 = 0
    var_1 = 1
    var_2 = 0.5

    res = Parallel(n_jobs=2, backend="loky")(
        delayed(generate_truncnorm_sbms)(
            m, block_1, block_2, mean_1, mean_2, var_1, var_2
        )
        for _ in range(2)
    )

    X1, Y1 = res[0]
    X2, Y2 = res[1]

    for i in range(m):
        assert ~np.all(X1[i] == X2[i])
        assert ~np.all(Y1[i] == Y2[i])
