import anndata as ad
import formulaic
import numpy as np
import pandas as pd

import pylemur.tl.lemur


def test_design_specification_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat)
    assert model.design_matrix.equals(formulaic.model_matrix("~ condition", dat))

    adata = ad.AnnData(Y)
    model = pylemur.tl.LEMUR(adata, design="~ condition", obs_data=dat)
    assert model.design_matrix.equals(formulaic.model_matrix("~ condition", dat))

    adata = ad.AnnData(Y, obs=dat)
    model = pylemur.tl.LEMUR(adata, design="~ condition")
    assert model.design_matrix.equals(formulaic.model_matrix("~ condition", dat))


def test_numpy_design_matrix_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    design_mat = formulaic.model_matrix("~ condition", dat).to_numpy()

    ref_model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat).fit()

    model = pylemur.tl.LEMUR(Y, design=design_mat).fit()
    assert np.allclose(model.coefficients, ref_model.coefficients)

    model = pylemur.tl.LEMUR(ad.AnnData(Y), design=design_mat).fit()
    assert np.allclose(model.coefficients, ref_model.coefficients)


def test_pandas_design_matrix_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    design_mat = formulaic.model_matrix("~ condition", dat)

    ref_model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat).fit()

    model = pylemur.tl.LEMUR(Y, design=design_mat).fit()
    assert np.allclose(model.coefficients, ref_model.coefficients)

    model = pylemur.tl.LEMUR(ad.AnnData(Y), design=design_mat).fit()
    assert np.allclose(model.coefficients, ref_model.coefficients)
