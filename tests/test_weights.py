import numpy as np
import pytest
import scipy.sparse as sp

from gpu_esda import inverse_distance_weights


def test_inverse_squared_dense(coordinates):
    w = inverse_distance_weights(
        coordinates, backend="cpu", output_format="dense", row_standardize=False
    )
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    expected = np.zeros_like(distances)
    np.divide(1, distances**2, out=expected, where=distances != 0)
    np.testing.assert_allclose(w, expected, atol=1e-10, rtol=1e-8)
    assert np.count_nonzero(np.diag(w)) == 0


def test_csr_row_standardized_and_diagnostics(coordinates):
    w, diag = inverse_distance_weights(
        coordinates, backend="cpu", output_format="csr", return_diagnostics=True
    )
    assert sp.isspmatrix_csr(w)
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), 1)
    assert diag.nnz == w.nnz and diag.islands == 0


def test_threshold_binary_and_island(coordinates):
    w, diag = inverse_distance_weights(
        coordinates,
        threshold=0.5,
        binary=True,
        output_format="csr",
        backend="cpu",
        return_diagnostics=True,
    )
    assert w.nnz == 0 and diag.islands == len(coordinates)


def test_knn_restriction(coordinates):
    w = inverse_distance_weights(
        coordinates, k=2, row_standardize=False, output_format="csr", backend="cpu"
    )
    assert np.all(np.diff(w.indptr) == 2)


def test_float32_and_auto(coordinates):
    w = inverse_distance_weights(coordinates, dtype="float32", backend="cpu")
    assert w.dtype == np.float32


@pytest.mark.parametrize("coords", [np.array([[0, 0], [0, 0]]), np.array([[0, np.nan], [1, 1]])])
def test_invalid_coordinates(coords):
    with pytest.raises(ValueError):
        inverse_distance_weights(coords, backend="cpu")
