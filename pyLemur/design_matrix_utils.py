import patsy
import numpy as np

def handle_design_parameter(design, data, obs_data):
    n_samples = data.shape[0]

    # Check if design is np.array
    if isinstance(design, np.ndarray):
        if design.ndim == 1:
            # Throw error
            raise ValueError("design specified as a 1d array is not supported yet")
        elif design.ndim == 2:
            design_matrix = design
            design_formula = None
        else:
            raise ValueError("design must be a 2d array")
    elif isinstance(design, str):
        design_matrix, design_formula = convert_formula_to_design_matrix(design, obs_data)
    else:
        raise ValueError("design must be a 2d array or string")

def convert_formula_to_design_matrix(formula, obs_data):
    # Check if formula is string
    if isinstance(formula, str):
        # Convert formula to design matrix
        design_matrix = patsy.dmatrix(formula, obs_data)
        return design_matrix, formula
    else:
        raise ValueError("formula must be a string")


def row_groups(matrix, return_reduced_matrix = False, return_group_ids = False):
    reduced_matrix, inv = np.unique(matrix, axis = 0, return_inverse=True)
    group_ids = np.unique(inv)
    if return_reduced_matrix and return_group_ids:
        return inv, reduced_matrix, group_ids
    elif return_reduced_matrix:
        return inv, reduced_matrix
    elif return_group_ids:
        return inv, group_ids
    else:
        return inv