import sklearn.decomposition as skd
from typing import NamedTuple
import numpy as np

class PCA(NamedTuple):
    embedding: np.ndarray
    coord_system: np.ndarray
    offset: np.ndarray

def fit_pca(Y, n, center=True):
    """
    Calculate the PCA of a given data matrix Y.
    Parameters
    ----------
    Y : array-like, shape (n_samples, n_features)
        The input data matrix.
    n : int
        The number of principal components to return.
    center : bool, default=True
        If True, the data will be centered before computing the covariance matrix.
    Returns
    -------
    pca : sklearn.decomposition.PCA
        The PCA object.
    """
    if center:
        pca = skd.PCA(n_components=n)
        emb = pca.fit_transform(Y)
        coord_system = pca.components_
        mean = pca.mean_
    else:
        svd = skd.TruncatedSVD(n_components=n)
        emb = svd.fit_transform(Y)
        coord_system =  svd.components_
        mean = np.zeros(Y.shape[1])
    return PCA(emb, coord_system, mean)