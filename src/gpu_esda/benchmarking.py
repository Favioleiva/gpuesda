"""Synthetic datasets and synchronized benchmark utilities."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from .backend import BackendName, memory_diagnostics, select_backend, synchronize


def synthetic_coordinates(n: int, kind: str = "random", seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.random((n, 2))
    if kind == "clustered":
        centers = rng.random((max(2, int(np.sqrt(n) / 2)), 2))
        groups = rng.integers(0, len(centers), n)
        return centers[groups] + rng.normal(0, 0.03, (n, 2))
    if kind == "grid":
        side = int(np.ceil(np.sqrt(n)))
        x, y = np.meshgrid(np.arange(side), np.arange(side))
        return np.column_stack((x.ravel(), y.ravel()))[:n].astype(float)
    raise ValueError("kind must be 'random', 'clustered', or 'grid'")


def synthetic_panel(
    n: int, variables: int = 1, periods: int = 1, seed: int = 12345, missing: bool = False
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, variables, periods))
    if missing:
        values[:: max(1, n // 10), 0, 0] = np.nan
    return values.squeeze() if variables == periods == 1 else values


@dataclass
class TimingStats:
    repetitions: int
    cold_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    mean_seconds: float
    std_seconds: float
    backend: str
    memory: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_call(
    function: Callable[[], Any], backend: BackendName = "cpu", repetitions: int = 5
) -> TimingStats:
    info = select_backend(backend)
    start = time.perf_counter()
    function()
    synchronize(info.name)
    cold = time.perf_counter() - start
    samples = []
    for _ in range(repetitions):
        synchronize(info.name)
        start = time.perf_counter()
        function()
        synchronize(info.name)
        samples.append(time.perf_counter() - start)
    values = np.asarray(samples)
    return TimingStats(
        repetitions,
        cold,
        float(np.median(values)),
        float(values.min()),
        float(values.max()),
        float(values.mean()),
        float(values.std()),
        info.name,
        memory_diagnostics(info.name),
    )
