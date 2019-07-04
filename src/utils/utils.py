import numpy as np


def n_to_labels(n):
    """Converts n vector (sbm input) to an array of labels
    
    Parameters
    ----------
    n : list or array
        length K vector indicating num vertices in each block
    
    Returns
    -------
    np.array
        shape (n_verts), indicator of what block each vertex 
        is in
    """
    n = np.array(n)
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels
