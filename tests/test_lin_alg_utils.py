import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from pylemur.tl._lin_alg_wrappers import *


def test_fit_pca():
    # Make example data
    Y = np.random.randn(30, 400)
    pca = fit_pca(Y, 3, center=True)
    assert np.allclose(pca.offset, Y.mean(axis=0))
    assert np.allclose(pca.coord_system @ pca.coord_system.T, np.eye(3))
    assert np.allclose(pca.embedding, (Y - pca.offset) @ pca.coord_system.T)

    pca2 = fit_pca(Y, 3, center=False)
    assert np.allclose(pca2.offset, np.zeros(Y.shape[1]))
    assert np.allclose(pca2.coord_system @ pca2.coord_system.T, np.eye(3))
    assert np.allclose(pca2.embedding, (Y - pca2.offset) @ pca2.coord_system.T)
    assert np.allclose(Y @ pca2.coord_system.T, pca2.embedding)


def test_ridge_regression():
    # Regular least squares
    Y = np.random.randn(400, 30)
    X = np.random.randn(400, 3)
    beta = ridge_regression(Y, X)
    assert beta.shape == (3, 30)
    assert np.allclose(beta, np.linalg.inv(X.T @ X) @ X.T @ Y)
    reg = LinearRegression(fit_intercept=False).fit(X, Y)
    assert np.allclose(beta, reg.coef_.T)

    # Check with weights
    weights = np.random.rand(400)
    beta = ridge_regression(Y, X, weights=weights)
    reg = LinearRegression(fit_intercept=False).fit(X, Y, sample_weight=weights)
    assert np.allclose(beta, reg.coef_.T)

    # Check with ridge penalty
    pen = 0.3
    beta = ridge_regression(Y, X, ridge_penalty=pen)
    gamma = np.sqrt(400) * pen**2 * np.eye(3)
    beta2 = np.linalg.inv(X.T @ X + gamma.T @ gamma) @ X.T @ Y
    assert np.allclose(beta, beta2)

    reg = Ridge(alpha=pen**4 * 400, fit_intercept=False).fit(X, Y)
    assert np.allclose(beta, reg.coef_.T)
