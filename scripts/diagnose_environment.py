"""Print a graceful, machine-readable CPU/GPU environment diagnostic."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from typing import Any


def _version(module: str) -> str | None:
    try:
        return importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        try:
            value = importlib.import_module(module)
            return str(getattr(value, "__version__", "installed"))
        except Exception:
            return None


def _nvidia_smi() -> dict[str, Any]:
    query = (
        "name,driver_version,memory.total,memory.free,"
        "compute_cap"
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        return {
            "name": fields[0],
            "driver_version": fields[1],
            "memory_total_mib": int(fields[2]),
            "memory_free_mib": int(fields[3]),
            "compute_capability": fields[4],
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def collect_diagnostics() -> dict[str, Any]:
    """Collect diagnostics without treating absent optional software as an error."""
    # CuPy otherwise defaults to a user profile that can be read-only in
    # sandboxed/managed environments.
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "gpu_esda_cupy"))
    try:
        import psutil

        vm = psutil.virtual_memory()
        cpu_model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
        host = {
            "cpu_model": cpu_model,
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "total_ram_bytes": vm.total,
            "available_ram_bytes": vm.available,
        }
    except Exception as exc:
        host = {"diagnostic_error": str(exc)}

    info: dict[str, Any] = {
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        **host,
        "packages": {
            name: _version(name)
            for name in (
                "numpy", "scipy", "pandas", "geopandas", "libpysal", "esda", "numba", "cupy"
            )
        },
        "nvidia_smi": _nvidia_smi(),
        "numba_cuda_available": False,
        "cupy": {"available": False},
    }
    try:
        from numba import cuda

        info["numba_cuda_available"] = bool(cuda.is_available())
    except Exception as exc:
        info["numba_cuda_error"] = str(exc)
    try:
        import cupy as cp

        device = cp.cuda.Device()
        free, total = device.mem_info
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        info["cupy"] = {
            "available": True,
            "version": cp.__version__,
            "device_id": device.id,
            "device_name": props["name"].decode(),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "memory_total_bytes": total,
            "memory_free_bytes": free,
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
        }
    except Exception as exc:
        info["cupy"] = {"available": False, "reason": str(exc)}
    return info


def main() -> int:
    print(json.dumps(collect_diagnostics(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
