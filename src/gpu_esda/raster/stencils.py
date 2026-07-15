"""Public raster-stencil definitions (implemented outside the raster import cycle)."""

from ..operators.stencils import (
    Stencil,
    inverse_distance_stencil,
    queen_stencil,
    rook_stencil,
)

__all__ = ["Stencil", "inverse_distance_stencil", "queen_stencil", "rook_stencil"]
