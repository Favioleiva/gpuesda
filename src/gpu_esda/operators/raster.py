"""Implicit NumPy/CuPy raster stencil operator."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..backend import BackendName, _cupy, select_backend
from ..raster.stencils import Stencil, inverse_distance_stencil, queen_stencil, rook_stencil
from .base import SpatialOperator

Normalization = Literal["row", "none"]


def _shift_slices(length: int, delta: int) -> tuple[slice, slice]:
    if delta >= 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(-delta, length), slice(0, length + delta)


class RasterStencilOperator(SpatialOperator):
    """Spatial weights represented only by mask, offsets, and local weights."""

    def __init__(
        self,
        mask: Any,
        stencil: Stencil,
        normalization: Normalization = "row",
        backend: BackendName = "auto",
        dtype: str = "float32",
    ) -> None:
        if normalization not in {"row", "none"}:
            raise ValueError("normalization must be 'row' or 'none'")
        self.backend = select_backend(backend).name
        self._xp = np if self.backend == "cpu" else _cupy()
        self.mask = self._xp.asarray(mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError("raster mask must be two-dimensional")
        self.shape = tuple(int(v) for v in self.mask.shape)
        self.stencil = stencil
        self.normalization = normalization
        self.dtype = self._xp.dtype(dtype)
        self._raw_row_sums = self._compute_raw_row_sums()
        if normalization == "row":
            self._row_sums = self._xp.where(
                self.mask & (self._raw_row_sums > 0), self._xp.asarray(1, dtype=self.dtype), 0
            )
        else:
            self._row_sums = self._raw_row_sums

    @property
    def xp(self) -> Any:
        return self._xp

    def _compute_raw_row_sums(self) -> Any:
        height, width = self.shape
        denominator = self._xp.zeros(self.shape, dtype=self.dtype)
        for (dr, dc), weight in zip(self.stencil.offsets, self.stencil.weights, strict=True):
            dest_r, src_r = _shift_slices(height, dr)
            dest_c, src_c = _shift_slices(width, dc)
            valid = self.mask[dest_r, dest_c] & self.mask[src_r, src_c]
            denominator[dest_r, dest_c] += valid * self.dtype.type(weight)
        return denominator

    def apply(self, values: Any) -> Any:
        native = self._xp.asarray(values)
        if tuple(native.shape[-2:]) != self.shape:
            raise ValueError(
                f"values end in {native.shape[-2:]}, expected raster shape {self.shape}"
            )
        if not bool(self._xp.all(self._xp.isfinite(native[..., self.mask])).item()):
            raise ValueError("valid raster cells contain NaN or infinity")
        clean = self._xp.where(self.mask, native, self._xp.asarray(0, dtype=native.dtype))
        output = self._xp.zeros_like(clean)
        height, width = self.shape
        for (dr, dc), weight in zip(self.stencil.offsets, self.stencil.weights, strict=True):
            dest_r, src_r = _shift_slices(height, dr)
            dest_c, src_c = _shift_slices(width, dc)
            dest = (..., dest_r, dest_c)
            src = (..., src_r, src_c)
            valid = self.mask[dest_r, dest_c] & self.mask[src_r, src_c]
            output[dest] += clean[src] * valid * native.dtype.type(weight)
        if self.normalization == "row":
            denominator = self._raw_row_sums
            safe = self._xp.where(denominator > 0, denominator, 1)
            output = self._xp.where(denominator > 0, output / safe, 0)
        return self._xp.where(self.mask, output, self._xp.asarray(0, dtype=output.dtype))

    def row_sums(self) -> Any:
        return self._row_sums

    def neighbor_weights(self) -> Any:
        """Return padded offset weights at valid focal cells for conditional inference."""
        height, width = self.shape
        table = self._xp.zeros((len(self.stencil.offsets), *self.shape), dtype=self.dtype)
        for index, ((dr, dc), weight) in enumerate(
            zip(self.stencil.offsets, self.stencil.weights, strict=True)
        ):
            dest_r, src_r = _shift_slices(height, dr)
            dest_c, src_c = _shift_slices(width, dc)
            valid = self.mask[dest_r, dest_c] & self.mask[src_r, src_c]
            table[index, dest_r, dest_c] = valid * self.dtype.type(weight)
        if self.normalization == "row":
            denominator = self._raw_row_sums[None, ...]
            safe = self._xp.where(denominator > 0, denominator, 1)
            table = self._xp.where(denominator > 0, table / safe, 0)
        return table


class RasterWeights:
    """Convenience constructors for implicit raster spatial weights."""

    @staticmethod
    def rook(
        mask: Any,
        normalization: Normalization = "row",
        backend: BackendName = "auto",
        dtype: str = "float32",
    ) -> RasterStencilOperator:
        return RasterStencilOperator(mask, rook_stencil(), normalization, backend, dtype)

    @staticmethod
    def queen(
        mask: Any,
        normalization: Normalization = "row",
        backend: BackendName = "auto",
        dtype: str = "float32",
    ) -> RasterStencilOperator:
        return RasterStencilOperator(mask, queen_stencil(), normalization, backend, dtype)

    @staticmethod
    def inverse_distance(
        mask: Any,
        radius: int = 2,
        normalization: Normalization = "row",
        backend: BackendName = "auto",
        dtype: str = "float32",
    ) -> RasterStencilOperator:
        return RasterStencilOperator(
            mask, inverse_distance_stencil(radius), normalization, backend, dtype
        )
