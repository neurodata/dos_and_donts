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


def compute_pr_at_k(pvalues, k, true_labels):
    """
    Computes precision and recall at various k. 

    Parameters
    ----------
    pvalues : 2d-array like, shape (n, n)
        P-values obtained from some test.
    k : int or array-like
        Values @k to compute precision and recall for. If list, compute P/R for
        each value in list. Otherwise, it computes P/R from range(1, k+1).
    true_labels : 1d-array with shape (n,)
        True community assignments.

    Returns
    -------
    precisions, recalls : array-like
        Computed precisions and recalls.
    """
    labels = true_labels.reshape(-1, 1)
    label_matrix = (labels @ labels.T) ^ 1

    triu_idx = np.triu_indices_from(pvalues, k=1)
    labels_vec = label_matrix[triu_idx]
    pvals_vec = pvalues[triu_idx]

    idx = np.argsort(pvals_vec)
    sorted_labels = labels_vec[idx]

    if isinstance(k, int):
        ks = range(1, k + 1)
    else:
        ks = k

    precisions = [sorted_labels[:k].mean() for k in ks]
    recalls = [sorted_labels[:k].sum() / sorted_labels.sum() for k in ks]

    return precisions, recalls