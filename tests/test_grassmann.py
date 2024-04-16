import numpy as np

from pylemur.tl._grassmann import *


def test_grassmann_map():
    # Test case 1: empty base point
    x = np.array([[1, 2], [3, 4], [5, 6]])
    p = np.array([])
    assert np.array_equal(grassmann_map(x, p), p)

    # Test case 2: x contains NaN values
    x = np.array([[1, 2], [3, np.nan], [5, 6]])
    p = np.array([[1, 0], [0, 1], [0, 0]])
    assert np.isnan(grassmann_map(x, p)).all()

    p = grassmann_random_point(5, 2)
    assert np.allclose(p.T @ p, np.eye(2))
    v = grassmann_random_tangent(p) * 10
    assert np.linalg.svd(v, compute_uv=False)[0] / np.pi * 180 > 90

    assert np.allclose(p.T @ v + v.T @ p, np.zeros((2, 2)))
    p2 = grassmann_map(v, p)
    valt = grassmann_log(p, p2)
    p3 = grassmann_map(valt, p)

    assert grassmann_angle_from_point(p3, p2) < 1e-10
    # assert np.linalg.matrix_rank(np.hstack([p3, p2])) == 2
    assert (valt**2).sum() < (v**2).sum()

    p4 = grassmann_random_point(5, 2)
    p5 = grassmann_random_point(5, 2)
    v45 = grassmann_log(p4, p5)
    assert np.allclose(p4.T @ v45 + v45.T @ p4, np.zeros((2, 2)))
    # This failed randomly on the github runner
    # assert np.linalg.matrix_rank(np.hstack([grassmann_map(v45, p4), p5])) == 2
    assert np.allclose(grassmann_log(p4, grassmann_map(v45, p4)), v45)


def test_experiment():
    assert 1 == 3 - 2
