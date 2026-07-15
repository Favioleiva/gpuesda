"""Common interface for explicit and implicit spatial weights."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SpatialOperator(ABC):
    """A spatial linear operator with explicit validity and row sums."""

    backend: str
    shape: tuple[int, ...]
    mask: Any

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array module (NumPy or CuPy)."""

    @abstractmethod
    def apply(self, values: Any) -> Any:
        """Apply the spatial lag without returning data to the host."""

    @abstractmethod
    def row_sums(self) -> Any:
        """Return effective row sums in the operator's native backend."""

    def s0(self) -> float:
        """Return the global sum of effective weights."""
        return float(self.row_sums().sum(dtype=self.xp.float64).item())

    def islands(self) -> Any:
        """Return valid positions with zero effective row sum."""
        return self.mask & (self.row_sums() == 0)
