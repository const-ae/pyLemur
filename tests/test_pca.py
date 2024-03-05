import numpy as np
from pyLemur.pca import *

def test_fit_pca():
    # Make example data
    Y = np.random.randn(30, 400)
    pca = fit_pca(Y, 3, center = True)
    assert np.allclose(pca.offset, Y.mean(axis = 0))
    assert np.allclose(pca.coord_system @ pca.coord_system.T, np.eye(3))
    assert np.allclose(pca.embedding, (Y - pca.offset) @ pca.coord_system.T)

    pca2 = fit_pca(Y, 3, center = False)
    assert np.allclose(pca2.offset, np.zeros(Y.shape[1]))
    assert np.allclose(pca2.coord_system @ pca2.coord_system.T, np.eye(3))
    assert np.allclose(pca2.embedding, (Y - pca2.offset) @ pca2.coord_system.T)

