from collections.abc import Mapping

import numpy as np
import pandas as pd

# import patsy
from formulaic import model_matrix
from numpy.lib import NumpyVersion


def handle_data(data, layer):
    Y = data.X if layer is None else data.layers[layer]
    if not isinstance(Y, np.ndarray):
        Y = Y.toarray()
    return Y


def handle_design_parameter(design, obs_data):
    if isinstance(design, np.ndarray):
        if design.ndim == 1:
            # Throw error
            raise ValueError("design specified as a 1d array is not supported yet")
        elif design.ndim == 2:
            design_matrix = pd.DataFrame(design)
            design_formula = None
        else:
            raise ValueError("design must be a 2d array")
    elif isinstance(design, pd.DataFrame):
        design_matrix = design
        design_formula = None
    elif isinstance(design, list):
        return handle_design_parameter(" * ".join(design), obs_data)
    elif isinstance(design, str):
        # Check if design starts with a ~
        if design[0] != "~":
            design = "~" + design + " - 1"
        design_matrix, design_formula = convert_formula_to_design_matrix(design, obs_data)
    else:
        raise ValueError("design must be a 2d array or string")

    return design_matrix, design_formula


def handle_obs_data(adata, obs_data):
    a = make_data_frame(adata.obs)
    b = make_data_frame(obs_data, preferred_index=a.index if a is not None else None)
    if a is None and b is None:
        return pd.DataFrame(index=pd.RangeIndex(0, adata.shape[0]))
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return pd.concat([a, b], axis=1)


def make_data_frame(data, preferred_index=None):
    if data is None:
        return None
    if isinstance(data, pd.DataFrame) and preferred_index is None:
        return data
    if isinstance(data, pd.DataFrame) and preferred_index is not None:
        if preferred_index.equals(data.index) or preferred_index.equals(data.index.map(str)):
            data.index = preferred_index
            return data
        else:
            raise ValueError("The index of adata.obs and obsData do not match")
    elif isinstance(data, Mapping):
        return pd.DataFrame(data, index=preferred_index)
    else:
        raise ValueError("data must be None, a pandas DataFrame or a Mapping object")


def convert_formula_to_design_matrix(formula, obs_data):
    # Check if formula is string
    if isinstance(formula, str):
        # Convert formula to design matrix
        # design_matrix = patsy.dmatrix(formula, obs_data)
        design_matrix = model_matrix(formula, obs_data)
        return design_matrix, formula
    else:
        raise ValueError("formula must be a string")


def row_groups(matrix, return_reduced_matrix=False, return_group_ids=False):
    reduced_matrix, inv = np.unique(matrix, axis=0, return_inverse=True)
    if NumpyVersion(np.__version__) >= "2.0.0rc":
        inv = np.squeeze(inv)
    group_ids = np.unique(inv)
    if return_reduced_matrix and return_group_ids:
        return inv, reduced_matrix, group_ids
    elif return_reduced_matrix:
        return inv, reduced_matrix
    elif return_group_ids:
        return inv, group_ids
    else:
        return inv
