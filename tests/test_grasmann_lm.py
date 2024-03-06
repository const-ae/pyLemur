
import numpy as np

from pyLemur.grassmann import grassmann_angle_from_point, grassmann_log, grassmann_map, grassmann_project
from pyLemur.grassmann_lm import grassmann_geodesic_regression, grassmann_lm
from pyLemur.lin_alg_wrappers import fit_pca


def test_geodesic_regression():
    n_feat = 5
    base_point = grassmann_project(np.random.randn(n_feat, 3)).T
    assert np.allclose(base_point @ base_point.T, np.eye(3))
    coord_systems = [grassmann_project(np.random.randn(n_feat, 3)).T for _ in range(10)]
    x = np.arange(10)
    design = np.vstack([np.ones(10), x]).T

    fit = grassmann_geodesic_regression(coord_systems, design, base_point)
    assert fit.shape == (3, n_feat, 2)
    proj = grassmann_map(fit[:,:,0].T, base_point.T)
    assert np.allclose(proj.T @ proj, np.eye(3))
    proj = grassmann_map(fit[:,:,1].T, base_point.T)
    assert np.allclose(proj.T @ proj, np.eye(3))


def test_grassmann_lm():
    n_obs = 100
    base_point = grassmann_project(np.random.randn(5, 2)).T
    data = np.random.randn(n_obs, 5)
    plane_all = fit_pca(data, 2, center = False).coord_system
    des = np.ones((n_obs, 1))
    fit = grassmann_lm(data, des, base_point)
    assert np.allclose(grassmann_angle_from_point(grassmann_map(fit[:,:,0].T, base_point.T), plane_all.T), 0)
    
    # Make a design matrix of three groups (with an intercept)
    x = np.random.randint(3, size=n_obs)
    des = np.eye(3)[x,:]
    des = np.hstack([np.ones((n_obs, 1)), des[:,1:3]])
    fit = grassmann_lm(data, des, base_point)

    plane_a = fit_pca(data[x == 0], 2, center = False).coord_system
    plane_b = fit_pca(data[x == 1], 2, center = False).coord_system
    plane_c = fit_pca(data[x == 2], 2, center = False).coord_system

    assert np.allclose(grassmann_angle_from_point(grassmann_map(fit[:,:,0].T, base_point.T), plane_a.T), 0)
    assert np.allclose(grassmann_angle_from_point(grassmann_map((fit[:,:,0] + fit[:,:,1]).T, base_point.T), plane_b.T), 0)
    assert np.allclose(grassmann_angle_from_point(grassmann_map((fit[:,:,0] + fit[:,:,2]).T, base_point.T), plane_c.T), 0)
