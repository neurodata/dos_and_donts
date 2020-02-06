import numpy as np
from graspy.cluster import GaussianCluster
from graspy.embed import MultipleASE, OmnibusEmbed
from graspy.models import SBMEstimator


def estimate_community(X, Y, true_labels, method, n_communities):
    """
    Function for estimating communities using embeddings methods.

    Parameters
    ----------
    X, Y : 3d-array like
        Input data. Must be of shape (m, n, n) for both.
    true_labels : 1d-array like
        True labels of the nodes. Must be length n.
    method : str
        Embedding method to be used. Must be {'mase', 'omni'}
    n_communities : int
        Number of communities to generate.

    Returns
    -------
    predicted_labels : 1d-array like
        Community assignment based on embedding and clustering.
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

    predicted_labels = GaussianCluster(n_communities, n_communities, "all").fit_predict(
        embeddings
    )

    # Label flipping
    idx = true_labels == 0
    if np.mean(predicted_labels[idx]) < 0.5:
        return predicted_labels
    else:
        # This is bitwise flipping. Turns all 0s to 1s and 1s to 0s.
        # Reason is to make labels consistent across repetitions
        predicted_labels = predicted_labels ^ (predicted_labels & 1 == predicted_labels)
        return predicted_labels


def estimate_block_probabilities(X, Y, labels):
    r"""
    Function to estimate \hat{b} for each sample.

    Parameters
    ----------
    X, Y : 3d-array like
        Input data. Must be of shape (m, n, n) for both.
    labels : 1d-array like
        Labels to use to estimate \hat{b}

    Returns
    -------
    bhat_1, bhat_2 : 3d-array like, shape (m, k, k)
        Estimated block probabilities for each input population.
    """
    bhat_1 = np.array([SBMEstimator(False).fit(x, labels).block_p_ for x in X])
    bhat_2 = np.array([SBMEstimator(False).fit(x, labels).block_p_ for x in Y])

    return bhat_1, bhat_2


def compute_pr_at_k(different_n, k, test_statistics):
    n = test_statistics.shape[0]
    labels = np.zeros((n, n))
    labels[0:different_n, 0:different_n] = 1

    triu_idx = np.triu_indices_from(test_statistics, k=1)
    test_statistics_ = np.abs(test_statistics[triu_idx])
    labels_ = labels[triu_idx]

    idx = np.argsort(test_statistics_)[::-1]
    sorted_labels = labels_[idx]

    precision_at_k = sorted_labels[:k].mean()
    recall_at_k = sorted_labels[:k].sum() / sorted_labels.sum()

    return precision_at_k, recall_at_k
