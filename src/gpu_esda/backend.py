"""Backend discovery, conversion, synchronization, and memory helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

import scipy.sparse as sp

BackendName = Literal["cpu", "gpu", "auto"]


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested backend cannot be used."""


@dataclass(frozen=True)
class BackendInfo:
    name: Literal["cpu", "gpu"]
    cupy_available: bool
    device_name: str | None = None
    reason: str | None = None


def _cupy() -> Any:
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "gpu_esda_cupy"))
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    return cp


def gpu_available() -> bool:
    """Return true only if CuPy imports and a CUDA device is usable."""
    try:
        return _cupy().cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def select_backend(backend: BackendName = "auto") -> BackendInfo:
    """Resolve a requested backend, falling back only for ``auto``."""
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("backend must be 'cpu', 'gpu', or 'auto'")
    if backend == "cpu":
        return BackendInfo("cpu", gpu_available())
    try:
        cp = _cupy()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return BackendInfo("gpu", True, props["name"].decode())
    except Exception as exc:
        if backend == "gpu":
            raise BackendUnavailableError(
                f"GPU backend requested but CUDA/CuPy is unavailable: {exc}"
            ) from exc
        return BackendInfo("cpu", False, reason=str(exc))


def is_gpu_array(value: Any) -> bool:
    try:
        cp = _cupy()
        import cupyx.scipy.sparse as cpsp

        return isinstance(value, (cp.ndarray, cpsp.spmatrix))
    except Exception:
        return False


def to_backend(value: Any, backend: BackendName = "auto", dtype: Any = None) -> Any:
    """Convert an array/sparse matrix to the selected native representation."""
    info = select_backend(backend)
    if info.name == "cpu":
        if is_gpu_array(value):
            return to_numpy(value).astype(dtype, copy=False) if dtype else to_numpy(value)
        return value.astype(dtype, copy=False) if dtype and hasattr(value, "astype") else value
    cp = _cupy()
    import cupyx.scipy.sparse as cpsp

    if sp.issparse(value):
        return cpsp.csr_matrix(value, dtype=dtype)
    if isinstance(value, cpsp.spmatrix):
        return value.astype(dtype, copy=False) if dtype else value
    return cp.asarray(value, dtype=dtype)


def to_numpy(value: Any) -> Any:
    """Return a NumPy/SciPy representation without exposing CuPy to callers."""
    try:
        cp = _cupy()
        import cupyx.scipy.sparse as cpsp

        if isinstance(value, cpsp.spmatrix):
            return value.get()
        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except Exception:
        pass
    return value


def synchronize(backend: BackendName = "auto") -> None:
    if select_backend(backend).name == "gpu":
        _cupy().cuda.Stream.null.synchronize()


def memory_diagnostics(backend: BackendName = "auto") -> dict[str, int | str | None]:
    info = select_backend(backend)
    if info.name == "cpu":
        import psutil

        vm = psutil.virtual_memory()
        return {"backend": "cpu", "free_bytes": vm.available, "total_bytes": vm.total}
    cp = _cupy()
    free, total = cp.cuda.Device().mem_info
    pool = cp.get_default_memory_pool()
    return {
        "backend": "gpu",
        "free_bytes": free,
        "total_bytes": total,
        "pool_used_bytes": pool.used_bytes(),
        "pool_total_bytes": pool.total_bytes(),
    }
