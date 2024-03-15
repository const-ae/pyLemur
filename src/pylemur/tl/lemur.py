
from typing import Any, Literal
from collections.abc import Iterable, Mapping
import numpy as np
import anndata as ad

from pyLemur.design_matrix_utils import *
from pyLemur.grassmann_lm import grassmann_lm, project_data_on_diffemb
from pyLemur.lin_alg_wrappers import *




def lemur(data: ad.AnnData, 
          design: str | list[str] | np.ndarray = "~ 1", 
          obs_data: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None, 
          n_embedding: int = 15, 
          linear_coefficient_estimator: Literal["linear", "zero"] = "linear",
          layer: str | None = None,
          copy: bool = True,
          verbose: bool = True):
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


    