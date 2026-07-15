"""Spatial operator abstractions for matrix and implicit raster weights."""

from .base import SpatialOperator
from .matrix import MatrixWeightsOperator
from .raster import RasterStencilOperator, RasterWeights

__all__ = ["MatrixWeightsOperator", "RasterStencilOperator", "RasterWeights", "SpatialOperator"]
