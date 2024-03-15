
import re
import pandas as pd
import numpy as np
import formulaic
import warnings
from pylemur.tl.alignment import reverse_linear_transformation

from pylemur.tl.design_matrix_utils import row_groups
from pylemur.tl.grassmann import grassmann_map


def predict(fit, 
            embedding = None,
            new_design = None,
            new_condition = None, 
            obs_data = None):
    if embedding is None:
        embedding = fit.obsm["embedding"]
    
    if new_condition is not None:
        if new_design is not None:
            warnings.warn("new_design is ignored if new_condition is provided.") 

        if isinstance(new_condition, pd.DataFrame):
            new_design = new_condition.to_numpy()
        elif isinstance(new_condition, np.ndarray):
            new_design = new_condition
        else:
            raise ValueError("new_condition must be a created using 'cond(...)' or a numpy array.")
        if new_design.shape[0] != 1:
            raise ValueError("new_condition must only have one row")
        # Repeat values row-wise
        new_design = np.ones((embedding.shape[0],1)) @ new_design
    
    # Make prediciton
    approx = new_design @ fit.uns["lemur"]["linear_coefficients"]

    coef = fit.uns["lemur"]["coefficients"]
    al_coefs = fit.uns["lemur"]["alignment_coefficients"]
    des_row_groups, reduced_design_matrix, des_row_group_ids = row_groups(new_design, return_reduced_matrix=True,return_group_ids=True)
    for id in des_row_group_ids:
        covars = reduced_design_matrix[id,:]
        subspace = grassmann_map(np.dot(coef, covars).T, fit.uns["lemur"]["base_point"].T)
        alignment = reverse_linear_transformation(al_coefs, covars)
        offset = np.dot(al_coefs[:,0,:], covars)
        approx[des_row_groups == id, :] += ((embedding[des_row_groups == id, :] - offset) @ alignment) @ subspace.T
    
    return approx


def cond(fit, **kwargs) -> np.ndarray:
    # This is copied from https://github.com/scverse/multi-condition-comparisions/blob/main/src/multi_condition_comparisions/tl/de.py#L164
    def _get_var_from_colname(colname):
        regex = re.compile(r"^.+\[T\.(.+)\]$")
        return regex.search(colname).groups()[0]

    design_matrix = fit.uns["lemur"]["design_matrix"]
    variables = design_matrix.model_spec.variables_by_source["data"]

    if not isinstance(design_matrix, formulaic.ModelMatrix):
        raise RuntimeError(
            "Building contrasts with `cond` only works if you specified the model using a "
            "formulaic formula. Please manually provide a contrast vector."
        )
    cond_dict = kwargs
    for var in variables:
        var_type = design_matrix.model_spec.encoder_state[var][0].value
        if var_type == "categorical":
            all_categories = set(design_matrix.model_spec.encoder_state[var][1]["categories"])
        if var in kwargs:
            if var_type == "categorical" and kwargs[var] not in all_categories:
                raise ValueError(
                    f"You specified a non-existant category for {var}. Possible categories: {', '.join(all_categories)}"
                )
        else:
            # fill with default values
            if var_type != "categorical":
                cond_dict[var] = 0
            else:
                var_cols = design_matrix.columns[design_matrix.columns.str.startswith(f"{var}[")]

                present_categories = {_get_var_from_colname(x) for x in var_cols}
                dropped_category = all_categories - present_categories
                assert len(dropped_category) == 1
                cond_dict[var] = next(iter(dropped_category))

    df = pd.DataFrame([kwargs])

    return design_matrix.model_spec.get_model_matrix(df)
