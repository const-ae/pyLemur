import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import anndata as ad
import formulaic_contrasts
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from pylemur.tl._design_matrix_utils import handle_data, handle_design_parameter, handle_obs_data, row_groups
from pylemur.tl._grassmann import grassmann_map
from pylemur.tl._grassmann_lm import grassmann_lm, project_data_on_diffemb
from pylemur.tl._lin_alg_wrappers import fit_pca, multiply_along_axis, ridge_regression
from pylemur.tl.alignment import (
    _align_impl,
    _apply_linear_transformation,
    _init_harmony,
    _reverse_linear_transformation,
)


class LEMUR:
    r"""Fit the LEMUR model

    A python implementation of the LEMUR algorithm. For more details please refer
    to Ahlmann-Eltze (2024).

    Parameters
    ----------
    data
        The AnnData object (or a different matrix container) with the variance stabilized data and the
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
        The name of the layer to use in `data`. If `None`, the `X` slot is used.
    copy
        Whether to make a copy of `data`.

    Attributes
    ----------
    embedding : :class:`~numpy.ndarray` (:math:`C \times P`)
        Low-dimensional representation of each cell
    adata : :class:`~anndata.AnnData`
        A reference to (potentially a copy of) the input data.
    data_matrix : :class:`~numpy.ndarray` (:math:`C \times G`)
        A reference to the data matrix from the `adata` object.
    n_embedding : int
        The number of latent dimensions
    design_matrix : `ModelMatrix` (:math:`C \times K`)
        The design matrix that is used for the fit.
    formula : str
        The design formula specification.
    coefficients : :class:`~numpy.ndarray` (:math:`P \times G \times K`)
        The 3D array of coefficients for the Grassmann regression.
    alignment_coefficients : :class:`~numpy.ndarray` (:math:`P \times (P+1) \times K`)
        The 3D array of coefficients for the affine alignment.
    linear_coefficients : :class:`~numpy.ndarray` (:math:`K\times G`)
        The 2D array of coefficients for the linear offset per condition.
    linear_coefficient_estimator : str
        The linear coefficient estimation specification.
    base_point :  :class:`~numpy.ndarray` (:math:`(P \times G`))
        The 2D array representing the reference subspace.

    Examples
    --------
    >>> model = pylemur.tl.LEMUR(adata, design="~ label + batch_cov", n_embedding=15)
    >>> model.fit()
    >>> model.align_with_harmony()
    >>> pred_expr = model.predict(new_condition=model.cond(label="treated"))
    >>> emb_proj = model_small.transform(adata)
    """

    def __init__(
        self,
        adata: ad.AnnData | Any,
        design: str | list[str] | np.ndarray = "~ 1",
        obs_data: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        n_embedding: int = 15,
        linear_coefficient_estimator: Literal["linear", "zero"] = "linear",
        layer: str | None = None,
        copy: bool = True,
    ):
        adata = _handle_data_arg(adata)
        if copy:
            adata = adata.copy()
        self.adata = adata

        adata.obs = handle_obs_data(adata, obs_data)
        design_matrix, formula = handle_design_parameter(design, adata.obs)

        if formula:
            self.contrast_builder = formulaic_contrasts.FormulaicContrasts(adata.obs, formula)
            assert design_matrix.equals(self.contrast_builder.design_matrix)
        else:
            self.contrast_builder = None

        self.design_matrix = design_matrix
        self.formula = formula
        if design_matrix.shape[0] != adata.shape[0]:
            raise ValueError("number of rows in design matrix must be equal to number of samples in data")
        self.data_matrix = handle_data(adata, layer)
        self.linear_coefficient_estimator = linear_coefficient_estimator
        self.n_embedding = n_embedding
        self.embedding = None
        self.coefficients = None
        self.linear_coefficients = None
        self.base_point = None
        self.alignment_coefficients = None

    def fit(self, verbose: bool = True):
        """Fit the LEMUR model

        Parameters
        ----------
        verbose
            Whether to print progress to the console.

        Returns
        -------
        `self`
            The fitted LEMUR model.
        """
        Y = self.data_matrix
        design_matrix = self.design_matrix
        n_embedding = self.n_embedding

        if self.linear_coefficient_estimator == "linear":
            if verbose:
                print("Centering the data using linear regression.")
            lin_coef = ridge_regression(Y, design_matrix.to_numpy())
            Y = Y - design_matrix.to_numpy() @ lin_coef
        else:  # linear_coefficient_estimator == "zero"
            lin_coef = np.zeros((design_matrix.shape[1], Y.shape[1]))

        if verbose:
            print("Find base point")
        base_point = fit_pca(Y, n_embedding, center=False).coord_system
        if verbose:
            print("Fit regression on latent spaces")
        coefficients = grassmann_lm(Y, design_matrix.to_numpy(), base_point)
        if verbose:
            print("Find shared embedding coordinates")
        embedding = project_data_on_diffemb(Y, design_matrix.to_numpy(), coefficients, base_point)

        embedding, coefficients, base_point = _order_axis_by_variance(embedding, coefficients, base_point)

        self.embedding = embedding
        self.alignment_coefficients = np.zeros((n_embedding, n_embedding + 1, design_matrix.shape[1]))
        self.coefficients = coefficients
        self.base_point = base_point
        self.linear_coefficients = lin_coef

        return self

    def align_with_harmony(
        self, ridge_penalty: float | list[float] | np.ndarray = 0.01, max_iter: int = 10, verbose: bool = True
    ):
        """Fine-tune the embedding with a parametric version of Harmony.

        Parameters
        ----------
        ridge_penalty
            The penalty controlling the flexibility of the alignment. Smaller
            values mean more flexible alignments.
        max_iter
            The maximum number of iterations to perform.
        verbose
            Whether to print progress to the console.


        Returns
        -------
        `self`
            The fitted LEMUR model with the updated embedding space stored in
            `model.embedding` attribute and an the updated alignment coefficients
            stored in `model.alignment_coefficients`.
        """
        if self.embedding is None:
            raise NotFittedError(
                "self.embedding is None. Make sure to call 'model.fit()' "
                + "before calling 'model.align_with_harmony()'."
            )

        embedding = self.embedding.copy()
        design_matrix = self.design_matrix
        # Init harmony
        harm_obj = _init_harmony(embedding, design_matrix, verbose=verbose)
        for idx in range(max_iter):
            if verbose:
                print(f"Alignment iteration {idx}")
            # Update harmony
            harm_obj.cluster()
            # alignment <- align_impl(training_fit$embedding, harm_obj$R, act_design_matrix, ridge_penalty = ridge_penalty)
            al_coef, new_emb = _align_impl(
                embedding,
                harm_obj.R,
                design_matrix,
                ridge_penalty=ridge_penalty,
                calculate_new_embedding=True,
                verbose=verbose,
            )
            harm_obj.Z_corr = new_emb.T
            harm_obj.Z_cos = multiply_along_axis(
                new_emb, 1 / np.linalg.norm(new_emb, axis=1).reshape((new_emb.shape[0], 1)), axis=1
            ).T

            if harm_obj.check_convergence(1):
                if verbose:
                    print("Converged")
                break
        self.alignment_coefficients = al_coef
        self.embedding = _apply_linear_transformation(embedding, al_coef, design_matrix)
        return self

    def align_with_grouping(
        self,
        grouping: list | np.ndarray | pd.Series,
        ridge_penalty: float | list[float] | np.ndarray = 0.01,
        preserve_position_of_NAs: bool = False,
        verbose: bool = True,
    ):
        """Fine-tune the embedding using annotated groups of cells.

        Parameters
        ----------
        grouping
            A list, :class:`~numpy.ndarray`, or pandas :class:`pandas.Series` specifying the group of cells.
            The groups span different conditions and can for example be cell types.
        ridge_penalty
            The penalty controlling the flexibility of the alignment.
        preserve_position_of_NAs
            `True` means that `NA`'s in the `grouping` indicate that these cells should stay
            where they are (if possible). `False` means that they are free to move around.
        verbose
            Whether to print progress to the console.

        Returns
        -------
        `self`
            The fitted LEMUR model with the updated embedding space stored in
            `model.embedding` attribute and an the updated alignment coefficients
            stored in `model.alignment_coefficients`.
        """
        if self.embedding is None:
            raise NotFittedError(
                "self.embedding is None. Make sure to call 'model.fit()' "
                + "before calling 'model.align_with_grouping()'."
            )
        embedding = self.embedding.copy()
        design_matrix = self.design_matrix
        if isinstance(grouping, list):
            grouping = pd.Series(grouping)

        if isinstance(grouping, pd.Series):
            grouping = grouping.factorize()[0] * 1.0
            grouping[grouping == -1] = np.nan

        al_coef = _align_impl(
            embedding,
            grouping,
            design_matrix,
            ridge_penalty=ridge_penalty,
            calculate_new_embedding=False,
            verbose=verbose,
        )
        self.alignment_coefficients = al_coef
        self.embedding = _apply_linear_transformation(embedding, al_coef, design_matrix)
        return self

    def transform(
        self,
        adata: ad.AnnData,
        layer: str | None = None,
        obs_data: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        return_type: Literal["embedding", "LEMUR"] = "embedding",
    ):
        """Transform data using the fitted LEMUR model

        Parameters
        ----------
        adata
            The AnnData object to transform.
        obs_data
            Optional set of annotations for each cell (same as `obs_data` in the
            constructor).
        return_type
            Flag that decides if the function returns a full `LEMUR` object or
            only the embedding.

        Returns
        -------
        :class:`~pylemur.tl.LEMUR`
            (if `return_type = "LEMUR"`) A new `LEMUR` object object with the embedding
            calculated for the input `adata`.

        :class:`~numpy.ndarray`
            (if `return_type = "embedding"`) A 2D numpy array of the embedding matrix
            calculated for the input `adata` (with cells in the rows and latent dimensions
            in the columns).
        """
        Y = handle_data(adata, layer)
        adata.obs = handle_obs_data(adata, obs_data)
        design_matrix, _ = handle_design_parameter(self.formula, adata.obs)
        dm = design_matrix.to_numpy()
        Y_clean = Y - dm @ self.linear_coefficients
        embedding = project_data_on_diffemb(
            Y_clean, design_matrix=dm, coefficients=self.coefficients, base_point=self.base_point
        )
        embedding = _apply_linear_transformation(embedding, self.alignment_coefficients, dm)
        if return_type == "embedding":
            return embedding
        elif return_type == "LEMUR":
            fit = LEMUR.copy()
            fit.adata = adata
            fit.design_matrix = design_matrix
            fit.embedding = embedding
            return fit

    def predict(
        self,
        embedding: np.ndarray | None = None,
        new_design: str | list[str] | np.ndarray | None = None,
        new_condition: np.ndarray | pd.DataFrame | None = None,
        obs_data: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        new_adata_layer: None | str = None,
    ):
        """Predict the expression of cells in a specific condition

        Parameters
        ----------
        embedding
            The coordinates of the cells in the shared embedding space. If None,
            the coordinates stored in `model.embedding` are used.
        new_design
            Either a design formula parsed using `model.adata.obs` and `obs_data` or
            a design matrix defining the condition for each cell. If both `new_design`
            and `new_condition` are None, the original design matrix
            (`model.design_matrix`) is used.
        new_condition
            A specification of the new condition that is applied to all cells. Typically,
            this is generated by `cond(...)`.
        obs_data
            A DataFrame-like object containing cell-wise annotations. It is only used if `new_design`
            contains a formulaic formula string.
        new_adata_layer
            If not `None`, the function returns `self` and stores the prediction in
            `model.adata["new_adata_layer"]`.

        Returns
        -------
        array-like, shape (n_cells, n_genes)
            The predicted expression of the cells in the new condition.
        """
        if embedding is None:
            if self.embedding is None:
                raise NotFittedError("The model has not been fitted yet.")
            embedding = self.embedding

        if new_condition is not None:
            if new_design is not None:
                warnings.warn("new_design is ignored if new_condition is provided.", stacklevel=1)

            if isinstance(new_condition, pd.DataFrame):
                new_design = new_condition.to_numpy()
            elif isinstance(new_condition, pd.Series):
                new_design = np.expand_dims(new_condition.to_numpy(), axis=0)
            elif isinstance(new_condition, np.ndarray):
                new_design = new_condition
            else:
                raise ValueError("new_condition must be a created using 'cond(...)' or a numpy array.")
            if new_design.shape[0] != 1:
                raise ValueError("new_condition must only have one row")
            # Repeat values row-wise
            new_design = np.ones((embedding.shape[0], 1)) @ new_design
        elif new_design is None:
            new_design = self.design_matrix.to_numpy()
        else:
            new_design = handle_design_parameter(new_design, handle_obs_data(self.adata, obs_data))[0].to_numpy()

        # Make prediciton
        approx = new_design @ self.linear_coefficients

        coef = self.coefficients
        al_coefs = self.alignment_coefficients
        des_row_groups, reduced_design_matrix, des_row_group_ids = row_groups(
            new_design, return_reduced_matrix=True, return_group_ids=True
        )
        for id in des_row_group_ids:
            covars = reduced_design_matrix[id, :]
            subspace = grassmann_map(np.dot(coef, covars).T, self.base_point.T)
            alignment = _reverse_linear_transformation(al_coefs, covars)
            offset = np.dot(al_coefs[:, 0, :], covars)
            approx[des_row_groups == id, :] += (
                (embedding[des_row_groups == id, :] - offset) @ alignment.T
            ) @ subspace.T
        if new_adata_layer is not None:
            self.adata.layers[new_adata_layer] = approx
            return self
        else:
            return approx

    def cond(self, **kwargs):
        """Define a condition for the `predict` function.

        Parameters
        ----------
        kwargs
            Named arguments specifying the levels of the covariates from the
            design formula. If a covariate is not specified, the first level is
            used.

        Returns
        -------
        `pd.Series`
            A contrast vector that aligns to the columns of the design matrix.


        Notes
        -----
        Subtracting two `cond(...)` calls, produces a contrast vector; these are
        commonly used in `R` to test for differences in a regression model.
        This pattern is inspired by the `R` package `glmGamPoi <https://bioconductor.org/packages/glmGamPoi/>`__.
        """
        if self.contrast_builder:
            return self.contrast_builder.cond(**kwargs)
        else:
            raise ValueError("The design was not specified as a formula. Cannot automatically construct contrast.")

    def __str__(self):
        if self.embedding is None:
            return f"LEMUR model (not fitted yet) with {self.n_embedding} dimensions"
        else:
            return f"LEMUR model with {self.n_embedding} dimensions"


def _handle_data_arg(data):
    if isinstance(data, ad.AnnData):
        return data
    else:
        return ad.AnnData(data)


def _order_axis_by_variance(embedding, coefficients, base_point):
    U, d, Vt = np.linalg.svd(embedding, full_matrices=False)
    base_point = Vt @ base_point
    coefficients = np.einsum("pq,qij->pij", Vt, coefficients)
    embedding = U @ np.diag(d)
    return embedding, coefficients, base_point
