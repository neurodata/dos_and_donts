import numpy as np
from graspy.simulations import er_np, sbm
from scipy.stats import truncnorm


def generate_binary_sbms(m, block_1, block_2, p, delta):
    """
    Function for generating two populations of undirected, binary SBMs.
    Population 1 is sampled with same `p` for all blocks. Population 2 is 
    sampled with `p + delta` for community 1, and with `p` for all other 
    communities. This function is used for Dos and Don'ts experiments 1.

    Parameters
    ----------
    m : int
        Number of samples per population
    block_1, block_2 : int
        Number of vertices in community 1 and 2, respectively.
    p : float
        Block matrix, or connectivity matrix, that defines connection
        probability within and across communities. 
    delta : float
        The effect size for a community 1.

    Returns
    -------
    pop1, pop2 : 3d-array with shape (m, n, n)
        Sampled undirected, binary graphs.
    labels : 1d-array with shape (n,)
        True community assignments.
    """
    total_n = block_1 + block_2
    n = [block_1, block_2]
    p2 = [[p + delta, p], [p, p]]

    pop1 = np.array([er_np(total_n, p, directed=False) for _ in np.arange(m)])
    pop2 = np.array([sbm(n, p2, directed=False) for _ in np.arange(m)])

    labels = np.array([0] * block_1 + [1] * block_2)

    return pop1, pop2, labels


def generate_binary_sbms_with_communities(m, block_1, block_2, p, q, delta):
    """
    Function for generating two populations of undirected, binary SBMs with 
    clear community structure. Population 1 is sampled with same `p` for all 
    blocks except block 2. Population 2 is sampled with `p + delta` for 
    community 1, and with `p` for all other communities. This function is 
    used for Dos and Don'ts experiments 3.

    Parameters
    ----------
    m : int
        Number of samples per population
    block_1, block_2 : int
        Number of vertices in community 1 and 2, respectively.
    p : float
        Block matrix, or connectivity matrix, that defines connection
        probability within and across communities. 
    q : float
        Block matrix, or connectivity matrix, that defines connection
        probability within and across communities. 
    delta : float
        The effect size for a community 1.

    Returns
    -------
    pop1, pop2 : 3d-array with shape (m, n, n)
        Sampled undirected, binary graphs.
    labels : 1d-array with shape (n,)
        True community assignments.
    """
    n = [block_1, block_2]
    p1 = [[p, p], [p, q]]
    p2 = [[p + delta, p], [p, q]]

    pop1 = np.array([sbm(n, p1, directed=False) for _ in np.arange(m)])
    pop2 = np.array([sbm(n, p2, directed=False) for _ in np.arange(m)])

    labels = np.array([0] * block_1 + [1] * block_2)

    return pop1, pop2, labels


def generate_truncnorm_sbms(m, block_1, block_2, mean_1, mean_2, var_1, var_2):
    """
    Function for generating two populations of undirected, weighted SBMs.
    The weight function is truncated normal such that all values are in [-1, 1].
    Population 1 is sampled with `mean_1` and `variance_1` for all blocks. Population 
    2 is sampled with `mean_2` and `variance_2` for community 1, and with `mean_1` 
    and `variance_1` for all other communities. This function is used for 
    Dos and Don'ts experiments 2 and 4.

    Parameters
    ----------
    m : int
        Number of samples per population
    block_1, block_2 : int
        Number of vertices in community 1 and 2, respectively.
    mean_1, mean_2 : float
        Means of truncated normal for community 1 and 2, respectively.
    var_1, var_2 : float
        Variances of truncated normal for community 1 and 2, respectively.

    Returns
    -------
    pop1, pop2 : 3d-array with shape (m, n, n)
        Sampled undirected, binary graphs.
    labels : 1d-array with shape (n,)
        True community assignments.
    """
    # Parameters for er and sbm functions
    total_n = block_1 + block_2
    n = [block_1, block_2]
    p = [[1, 1], [1, 1]]
    sd_1 = np.sqrt(var_1)
    sd_2 = np.sqrt(var_2)

    # deal with clip values
    a_1 = (-1 - mean_1) / sd_1
    b_1 = (1 - mean_1) / sd_1
    a_2 = (-1 - mean_2) / sd_2
    b_2 = (1 - mean_2) / sd_2

    pop_1 = []
    pop_2 = []
    for _ in range(m):
        # seeds are needed for joblib and scipy random functions
        # numpy random is not affected by joblib
        seeds = np.random.randint(0, 2147483647, size=4)

        wt_func = [[truncnorm.rvs, truncnorm.rvs], [truncnorm.rvs, truncnorm.rvs]]
        wt_args_1 = (dict(a=a_1, b=b_1, loc=mean_1, scale=sd_1, random_state=seeds[0]),)
        wt_args_2 = [
            [
                dict(a=a_2, b=b_2, loc=mean_2, scale=sd_1, random_state=seeds[1]),
                dict(a=a_1, b=b_1, loc=mean_1, scale=sd_1, random_state=seeds[2]),
            ],
            [
                dict(a=a_1, b=b_1, loc=mean_1, scale=sd_1, random_state=seeds[2]),
                dict(a=a_1, b=b_1, loc=mean_1, scale=sd_1, random_state=seeds[3]),
            ],
        ]

        pop_1.append(
            er_np(total_n, 1.0, directed=False, wt=truncnorm.rvs, wtargs=wt_args_1)
        )
        pop_2.append(sbm(n, p, directed=False, wt=wt_func, wtargs=wt_args_2))

    labels = np.array([0] * block_1 + [1] * block_2)

    return np.array(pop_1), np.array(pop_2), labels
