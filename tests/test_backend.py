import numpy as np
import pytest

import gpu_esda.backend as backend


def test_cpu_and_auto_detection():
    assert backend.select_backend("cpu").name == "cpu"
    assert backend.select_backend("auto").name in {"cpu", "gpu"}


def test_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        backend.select_backend("magic")


def test_explicit_gpu_failure(monkeypatch):
    def fail():
        raise ImportError("test unavailable")

    monkeypatch.setattr(backend, "_cupy", fail)
    with pytest.raises(backend.BackendUnavailableError, match="unavailable"):
        backend.select_backend("gpu")
    assert backend.select_backend("auto").name == "cpu"


def test_cpu_conversion_and_sync():
    value = np.arange(3)
    assert backend.to_backend(value, "cpu") is value
    np.testing.assert_array_equal(backend.to_numpy(value), value)
    backend.synchronize("cpu")
