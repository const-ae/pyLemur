import harmonypy
import numpy as np

from pyLemur.design_matrix_utils import row_groups
from pyLemur.lin_alg_wrappers import multiply_along_axis, ridge_regression


def align_with_harmony(fit, ridge_penalty = 0.01, max_iter = 10, verbose = True):

    embedding = fit.obsm["embedding"].copy()
    design_matrix = fit.uns["lemur"]["design_matrix"]
    # Init harmony
    harm_obj = init_harmony(embedding, design_matrix, verbose = verbose)
    for idx in range(max_iter):
        if verbose: print(f"Iteration {idx}")
        # Update harmony
        harm_obj.cluster()
        # alignment <- align_impl(training_fit$embedding, harm_obj$R, act_design_matrix, ridge_penalty = ridge_penalty)
        al_coef, new_emb = align_impl(embedding, harm_obj.R, design_matrix, ridge_penalty = ridge_penalty, calculate_new_embedding = True)
        harm_obj.Z_corr = new_emb.T
        harm_obj.Z_cos = multiply_along_axis(new_emb, 1 / np.linalg.norm(new_emb, axis=1).reshape((new_emb.shape[0], 1)), axis = 1).T

        if harm_obj.check_convergence(1):
            if verbose: print("Converged")
            break
    
    fit.uns["lemur"]["alignment_coefficients"] = al_coef
    fit.obsm["embedding"] = apply_linear_transformation(embedding, al_coef, design_matrix)
    return fit


def align_impl(embedding, grouping, design_matrix, ridge_penalty = 0.01, calculate_new_embedding = True):
    if grouping.ndim == 1:
        raise ValueError("grouping must be a 2d array")
    else:
        col_sums = grouping.sum(axis=0)
        col_sums[col_sums == 0] = 1
        grouping_matrix = grouping / col_sums
    
    # I could do something with NA's but the R code looks a bit complicated
    
    n_groups = grouping.shape[0]
    n_emb = embedding.shape[1]
    K = design_matrix.shape[1]

    des_row_groups, des_row_group_ids = row_groups(design_matrix, return_group_ids=True)
    n_conditions = des_row_group_ids.shape[0]
    cond_ct_means = [np.zeros((n_emb, n_groups)) for _ in des_row_group_ids]
    for id in des_row_group_ids:
        sel = des_row_groups == id
        for idx in range(n_groups):
            cond_ct_means[id][:,idx] = np.average(embedding[sel,:], axis=0, weights=grouping_matrix[idx,sel])
    
    target = np.zeros((n_emb, n_groups))
    for idx in range(n_groups):
        tmp = np.zeros((n_conditions, n_emb))
        for id in des_row_group_ids:
            tmp[id,:] = cond_ct_means[id][:,idx]
        target[:,idx] = tmp.sum(axis=0)

    new_pos = embedding.copy()
    for id in des_row_group_ids:
        sel = des_row_groups==id
        diff = target - cond_ct_means[id]
        new_pos[sel,:] = new_pos[sel,:] + (diff @ grouping_matrix[:,sel]).T

    intercept_emb = np.hstack([np.ones((embedding.shape[0],1)), embedding])
    interact_design_matrix = np.repeat(design_matrix, n_emb + 1, axis = 1) * np.hstack([intercept_emb] * K)
    alignment_coefs = ridge_regression(new_pos - embedding, interact_design_matrix, ridge_penalty)
    print(f"error: {np.linalg.norm((new_pos - embedding) - interact_design_matrix @ alignment_coefs)}")
    alignment_coefs = alignment_coefs.reshape((K, n_emb + 1, n_emb)).transpose((2, 1, 0))
    if calculate_new_embedding:
        new_embedding = apply_linear_transformation(embedding, alignment_coefs, design_matrix)
        return alignment_coefs, new_embedding
    else:
        return alignment_coefs
    

def apply_linear_transformation(embedding, alignment_coefs, design_matrix):
    des_row_groups, reduced_design_matrix, des_row_group_ids = row_groups(design_matrix, return_reduced_matrix=True,return_group_ids=True)
    embedding = embedding.copy()
    for id in des_row_group_ids:
        sel = des_row_groups == id
        embedding[sel,:] = np.hstack([np.ones((np.sum(sel),1)), embedding[sel,:]]) @ \
            forward_linear_transformation(alignment_coefs, reduced_design_matrix[id,:]).T              
    return embedding

def forward_linear_transformation(alignment_coef, design_vector):
    n_emb = alignment_coef.shape[0]
    if n_emb == 0:
        return np.zeros((0, 0))
    else:
        return np.hstack([np.zeros((n_emb, 1)), np.eye(n_emb)]) + np.dot(alignment_coef, design_vector)

def init_harmony(embedding, design_matrix, 
                 theta = 2,
                 lamb = 1,
                 sigma = 0.1, 
                 nclust = None,
                 tau = 0,
                 block_size = 0.05, 
                 max_iter_kmeans = 20,
                 epsilon_cluster = 1e-5,
                 epsilon_harmony = 1e-4, 
                 verbose = True):
    n_obs = embedding.shape[0]
    des_row_groups, des_row_group_ids = row_groups(design_matrix, return_group_ids=True)
    n_groups = len(des_row_group_ids)
    if nclust is None:
        nclust = np.min([np.round(n_obs / 30.0), 100]).astype(int)

    phi = np.eye(n_groups)[:,des_row_groups]
    phi_n = np.ones(n_groups)

    N_b = phi.sum(axis=1)
    Pr_b = N_b / n_obs
    sigma = np.repeat(sigma, nclust)

    theta = np.repeat(theta, n_groups)
    theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))
    
    lamb = np.repeat(lamb, n_groups)
    lamb_mat = np.diag(np.insert(lamb, 0, 0))
    phi_moe = np.vstack((np.repeat(1, n_obs), phi))

    max_iter_harmony = 0
    ho = harmonypy.Harmony(
        embedding.T, phi, phi_moe, Pr_b, sigma, theta, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, lamb_mat, verbose
    )
    return ho
