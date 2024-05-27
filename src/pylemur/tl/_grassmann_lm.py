import numpy as np

from pylemur.tl._design_matrix_utils import row_groups
from pylemur.tl._grassmann import grassmann_log, grassmann_map
from pylemur.tl._lin_alg_wrappers import fit_pca, ridge_regression


def grassmann_geodesic_regression(coord_systems, design, base_point, weights=None):
    """
    Fit geodesic regression on Grassmann manifolds

    Solve Sum_j d(U_j, Exp_p(Sum_k V_k:: * X_jk)) for V, where
    d(U, V) = ||Log(U, V)|| is the inverse of the exponential map on the Grassmann manifold.

    Parameters
    ----------
    coord_systems : a list of orthonormal 2D matrices (length n_groups)
    design : design matrix, shape (n_groups, n_coef)
    base_point : array-like, shape (n_emb, n_features)
        The base point.
    weights : array-like, shape (n_groups,)

    Returns
    -------
    beta: array-like, shape (n_emb, n_features, n_coef)
    """
    n_obs = design.shape[0]
    n_coef = design.shape[1]
    n_emb = base_point.shape[0]
    n_features = base_point.shape[1]

    assert len(coord_systems) == n_obs
    for i in range(n_obs):
        assert coord_systems[i].shape == (n_emb, n_features)
    if weights is None:
        weights = np.ones(n_obs)

    tangent_vecs = [grassmann_log(base_point.T, coord_systems[i].T).T.reshape(n_emb * n_features) for i in range(n_obs)]
    tangent_vecs = np.vstack(tangent_vecs)
    if tangent_vecs.shape[0] == 0:
        tangent_fit = np.zeros((0, n_coef))
    else:
        tangent_fit = ridge_regression(tangent_vecs, design, weights=weights)

    tangent_fit = tangent_fit.reshape((n_coef, n_emb, n_features)).transpose((1, 2, 0))
    return tangent_fit


def grassmann_lm(Y, design_matrix, base_point):
    """
    Solve Sum_i||Y_i: - Y_i: Proj(Exp_p(Sum_k V_k:: * X_ik))||^2 for V.

    Parameters
    ----------
    Y : array-like, shape (n_samples, n_features)
        The input data matrix.
    design_matrix : array-like, shape (n_samples, n_coef)
        The design matrix.
    base_point : array-like, shape (n_emb, n_features)
        The base point.

    Returns
    -------
    beta: array-like, shape (n_emb, n_features, n_coef)
    """
    n_emb = base_point.shape[0]

    des_row_groups, reduced_design_matrix, des_row_group_ids = row_groups(
        design_matrix, return_reduced_matrix=True, return_group_ids=True
    )
    if np.min(np.unique(des_row_groups, return_counts=True)[1]) < n_emb:
        raise ValueError("Too few dataset points in some design matrix group.")
    group_planes = [fit_pca(Y[des_row_groups == i, :], n_emb, center=False).coord_system for i in des_row_group_ids]
    group_sizes = [np.sum(des_row_groups == i) for i in des_row_group_ids]

    coef = grassmann_geodesic_regression(group_planes, reduced_design_matrix, base_point, weights=group_sizes)
    return coef


def project_diffemb_into_data_space(embedding, design_matrix, coefficients, base_point):
    n_features = base_point.shape[1]
    res = np.zeros((design_matrix.shape[0], n_features))
    des_row_groups, reduced_design_matrix, des_row_group_ids = row_groups(
        design_matrix, return_reduced_matrix=True, return_group_ids=True
    )
    for id in des_row_group_ids:
        covar = reduced_design_matrix[id, :]
        subspace = grassmann_map(np.dot(coefficients, covar).T, base_point.T)
        res[des_row_groups == id, :] = embedding[des_row_groups == id, :] @ subspace.T
    return res


def project_data_on_diffemb(Y, design_matrix, coefficients, base_point):
    n_emb = base_point.shape[0]
    n_obs = Y.shape[0]
    res = np.zeros((n_obs, n_emb))
    des_row_groups, reduced_design_matrix, des_row_group_ids = row_groups(
        design_matrix, return_reduced_matrix=True, return_group_ids=True
    )
    for id in des_row_group_ids:
        Y_subset = Y[des_row_groups == id, :]
        covar = reduced_design_matrix[id, :]
        subspace = grassmann_map(np.dot(coefficients, covar).T, base_point.T)
        res[des_row_groups == id, :] = Y_subset @ subspace
    return res
