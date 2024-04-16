from typing import NamedTuple

import numpy as np
import sklearn.decomposition as skd


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
        svd = skd.TruncatedSVD(n_components=n, algorithm="arpack")
        emb = svd.fit_transform(Y)
        coord_system = svd.components_
        mean = np.zeros(Y.shape[1])
    return PCA(emb, coord_system, mean)


def ridge_regression(Y, X, ridge_penalty=0, weights=None):
    """
    Calculate the ridge regression of a given data matrix Y.

    Parameters
    ----------
    Y : array-like, shape (n_samples, n_features)
        The input data matrix.
    X : array-like, shape (n_samples, n_coef)
        The input data matrix.
    ridge_penalty : float, default=0
        The ridge penalty.
    weights : array-like, shape (n_features,)
        The weights to apply to each feature.

    Returns
    -------
    ridge: array-like, shape (n_coef, n_features)
    """
    n_coef = X.shape[1]
    n_samples = X.shape[0]
    n_feat = Y.shape[1]
    assert Y.shape[0] == n_samples
    if weights is None:
        weights = np.ones(n_samples)
    assert len(weights) == n_samples

    if np.ndim(ridge_penalty) == 0 or len(ridge_penalty) == 1:
        ridge_penalty = np.eye(n_coef) * ridge_penalty
    elif np.ndim(ridge_penalty) == 1:
        assert len(ridge_penalty) == n_coef
        ridge_penalty = np.diag(ridge_penalty)
    elif np.ndim(ridge_penalty) == 1:
        assert ridge_penalty.shape == (n_coef, n_coef)
        pass
    else:
        raise ValueError("ridge_penalty must be a scalar, 1d array, or 2d array")

    ridge_penalty_sq = np.sqrt(np.sum(weights)) * (ridge_penalty.T @ ridge_penalty)
    weights_sqrt = np.sqrt(weights)
    X_ext = np.vstack([multiply_along_axis(X, weights_sqrt, 0), ridge_penalty_sq])
    Y_ext = np.vstack([multiply_along_axis(Y, weights_sqrt, 0), np.zeros((n_coef, n_feat))])

    ridge = np.linalg.lstsq(X_ext, Y_ext, rcond=None)[0]
    return ridge


def multiply_along_axis(A, B, axis):
    # Copied from https://stackoverflow.com/a/71750176/604854
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)
