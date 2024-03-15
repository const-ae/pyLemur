
from typing import Any, Literal, Union
from collections.abc import Iterable, Mapping
import numpy as np
import anndata as ad

from pylemur.tl._design_matrix_utils import *
from pylemur.tl._grassmann_lm import grassmann_lm, project_data_on_diffemb
from pylemur.tl._lin_alg_wrappers import *




def lemur(data: ad.AnnData, 
          design: Union[str, list[str], np.ndarray] = "~ 1", 
          obs_data: Union[pd.DataFrame, Mapping[str, Iterable[Any]], None] = None, 
          n_embedding: int = 15, 
          linear_coefficient_estimator: Literal["linear", "zero"] = "linear",
          layer: Union[str, None] = None,
          copy: bool = True,
          verbose: bool = True):
    """Fit the LEMUR model

    Parameters
    ----------
    data
        The AnnData object containing the variance stabilized data and the
        cell-wise annotations in `data.obs`.
    design
        A specification of the experimental design. This can be a string, 
        which is then parsed using `formulaic`. Alternatively, it can be a
        a list of strings, which are assumed to refer to the columns in
        `data.obs`. Finally, it can be a numpy array, representing a 
        design matrix of size `n_cells` x `n_covariates`. If not provided,
        a constant design is used.
    obs_data
        A pandas DataFrame or a dictionary of iterables containing the 
        cell-wise annotations. It is used in combination with the 
        information in `data.obs`.
    n_embedding
        The number of dimensions to use for the shared embedding space. 
    linear_coefficient_estimator
        The method to use for estimating the linear coefficients. If `"linear"`,
        the linear coefficients are estimated using ridge regression. If `"zero"`,
        the linear coefficients are set to zero.
    layer
        The name of the layer to use in `data`. If None, the X slot is used.
    copy
        Whether to make a copy of `data`.
    verbose
        Whether to print progress to the console.

    Returns
    -------
    ad.AnnData
        The input AnnData object with the shared embedding space stored in 
        `data.obsm["embedding"]` and the LEMUR coefficients stored in 
        `data.uns["lemur"]`.
    """
    if copy:
        data = data.copy()
    
    data.obs = handle_obs_data(data, obs_data)
    design_matrix, formula = handle_design_parameter(design, data.obs)
    if design_matrix.shape[0] != data.shape[0]:
        raise ValueError("number of rows in design matrix must be equal to number of samples in data")

    Y = handle_data(data, layer)

    
    if linear_coefficient_estimator == "linear":
        if verbose:
            print("Centering the data using linear regression.")
        lin_coef = ridge_regression(Y, design_matrix.to_numpy())
        Y = Y - design_matrix.to_numpy() @ lin_coef
    else: # linear_coefficient_estimator == "zero"
        lin_coef = np.zeros((design_matrix.shape[1], Y.shape[1]))

    if verbose:
        print("Find base point")
    base_point = fit_pca(Y, n_embedding, center = False).coord_system
    if verbose:
        print("Fit regression on latent spaces")
    coefficients = grassmann_lm(Y, design_matrix.to_numpy(), base_point)
    if verbose:
        print("Find shared embedding coordinates")
    embedding = project_data_on_diffemb(Y, design_matrix.to_numpy(), coefficients, base_point)

    embedding, coefficients, base_point = order_axis_by_variance(embedding, coefficients, base_point)

    data.obsm["embedding"] = embedding
    al_coef = np.zeros((n_embedding, n_embedding + 1, design_matrix.shape[1]))
    data.uns["lemur"] = {"coefficients": coefficients, 
                         "alignment_coefficients": al_coef,
                         "base_point": base_point,
                         "formula": formula,
                         "design_matrix": design_matrix,
                         "n_embedding": n_embedding,
                         "linear_coefficients": lin_coef}
    return data

def order_axis_by_variance(embedding, coefficients, base_point):
    U, d, Vt = np.linalg.svd(embedding, full_matrices=False)
    base_point = Vt @ base_point
    coefficients = np.einsum('pq,qij->pij', Vt, coefficients)
    embedding = U @ np.diag(d)
    return embedding, coefficients, base_point


    