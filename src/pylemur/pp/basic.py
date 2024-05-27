import numpy as np
import scipy


def shifted_log_transform(counts, overdispersion=0.05, pseudo_count=None, minimum_overdispersion=0.001):
    r"""Apply log transformation to count data

    The transformation is proportional to :math:`\log(y/s + c)`, where :math:`y` are the counts, :math:`s` is the size factor,
    and :math:`c` is the pseudo-count.

    The actual transformation is :math:`a \, \log(y/(sc) + 1)`, where using :math:`+1` ensures that the output remains
    sparse for :math:`y=0` and the scaling with :math:`a=\sqrt{4c}` ensures that the transformation approximates the :math:`\operatorname{acosh}`
    transformation. Using :math:`\log(y/(sc) + 1)` instead of :math:`\log(y/s+c)`
    only changes the results by a constant offset, as :math:`\log(y+c) = \log(y/c + 1) - \log(1/c)`. Importantly, neither
    scaling nor offsetting by a constant changes the variance-stabilizing quality of the transformation.

    The size factors are calculated as normalized sum per cell::

        size_factors = counts.sum(axis=1)
        size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))

    In case `y` is not a matrix, the `size_factors` are fixed to 1.

    Parameters
    ----------
    counts
        The count matrix which can be sparse.
    overdispersion,pseudo_count
        Specification how much variation is expected for a homogeneous sample. The `overdispersion` and
        `pseudo_count` are related by `overdispersion = 1 / (4 * pseudo_count)`.
    minimum_overdispersion
        Avoid overdispersion values smaller than `minimum_overdispersion`.

    Returns
    -------
    A matrix of variance-stabilized values.
    """
    if pseudo_count is None:
        pseudo_count = 1 / (4 * overdispersion)

    n_cells = counts.shape[0]
    size_factors = counts.sum(axis=1)
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))
    norm_mat = counts / size_factors.reshape((n_cells, 1))
    overdispersion = 1 / (4 * pseudo_count)
    if overdispersion < minimum_overdispersion:
        overdispersion = minimum_overdispersion
    res = 1 / np.sqrt(overdispersion) * np.log1p(4 * overdispersion * norm_mat)
    if scipy.sparse.issparse(counts):
        res = scipy.sparse.csr_matrix(res)
    return res
