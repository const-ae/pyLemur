import numpy as np
import scanpy.preprocessing._simple
import scipy


def shifted_log_transform(counts, overdispersion=0.05, pseudo_count=None, minimum_overdispersion=0.001):
    if pseudo_count is None:
        pseudo_count = 1 / (4 * overdispersion)

    n_cells = counts.shape[0]
    size_factors = counts.sum(axis=1)
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))
    norm_mat = counts / size_factors.reshape((n_cells, 1))
    overdispersion = 1 / (4 * pseudo_count)
    res = 1 / np.sqrt(overdispersion) * np.log1p(4 * overdispersion * norm_mat)
    if scipy.sparse.issparse(counts):
        res = scipy.sparse.csr_matrix(res)
    return res


def get_top_hvgs(adata, n=1000, layer=None):
    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]
    var = scanpy.preprocessing._simple._get_mean_var(mat)[1]
    return var.argsort()[: -(n + 1) : -1]
