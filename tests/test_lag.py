import libpysal
import numpy as np
import scipy.sparse as sp

from gpu_esda import spatial_lag


def test_vector_dense_and_csr(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    expected = libpysal.weights.lag_spatial(w, values)
    np.testing.assert_allclose(spatial_lag(w.sparse.toarray(), values, "cpu"), expected)
    np.testing.assert_allclose(spatial_lag(sp.csr_matrix(w.sparse), values, "cpu"), expected)


def test_matrix_and_panel_shape(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    matrix = np.column_stack([values, values**2])
    np.testing.assert_allclose(spatial_lag(w.sparse, matrix, "cpu"), w.sparse @ matrix)
    panel = np.stack([matrix, matrix + 1], axis=2)
    assert spatial_lag(w.sparse, panel, "cpu").shape == panel.shape


def test_float32(values):
    eye = np.eye(5, dtype=np.float32)
    result = spatial_lag(eye, values.astype(np.float32), "cpu")
    assert result.dtype == np.float32
