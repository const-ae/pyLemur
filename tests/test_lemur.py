from io import StringIO

import anndata as ad
import formulaic
import numpy as np
import pandas as pd

import pylemur.tl._grassmann
import pylemur.tl.alignment
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


def test_copy_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat)
    cp = model.copy(copy_adata=False)
    cp2 = model.copy(copy_adata=True)
    assert id(model.adata) == id(cp.adata)
    assert id(model.adata) != id(cp2.adata)
    _assert_lemur_model_equal(model, cp)
    _assert_lemur_model_equal(model, cp2, adata_id_equal=False)
    


def _assert_lemur_model_equal(m1, m2, adata_id_equal = True):
    for k in m1.__dict__.keys():
        if k == "adata" and adata_id_equal:
            assert id(m1.adata) == id(m2.adata)
        elif k == "adata" and not adata_id_equal:
            assert id(m1.adata) != id(m2.adata)
        elif isinstance(m1.__dict__[k], pd.DataFrame):
            pd.testing.assert_frame_equal(m1.__dict__[k], m2.__dict__[k])
        elif isinstance(m1.__dict__[k], np.ndarray):
            assert np.array_equal(m1.__dict__[k], m2.__dict__[k])
        else:
            assert m1.__dict__[k] == m2.__dict__[k]