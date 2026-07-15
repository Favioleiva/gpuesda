"""Raster data, stencil definitions, statistics, and permutation inference."""

from .grid import BlackMarbleRaster
from .statistics import (
    RasterMoranGlobalResult,
    RasterMoranLocalResult,
    moran_global,
    moran_local,
    moran_observed,
)
from .stencils import Stencil, inverse_distance_stencil, queen_stencil, rook_stencil

__all__ = [
    "BlackMarbleRaster",
    "RasterMoranGlobalResult",
    "RasterMoranLocalResult",
    "Stencil",
    "inverse_distance_stencil",
    "moran_global",
    "moran_local",
    "moran_observed",
    "queen_stencil",
    "rook_stencil",
]
