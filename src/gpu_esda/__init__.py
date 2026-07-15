"""Public API for GPU Powered ESDA."""

from .backend import BackendUnavailableError, gpu_available, memory_diagnostics, select_backend
from .lag import spatial_lag
from .local_moran import MoranLocal, local_moran_panel
from .moran import Moran, moran_panel
from .multiple_testing import adjust_pvalues
from .weights import inverse_distance_weights

__all__ = [
    "BackendUnavailableError",
    "Moran",
    "MoranLocal",
    "adjust_pvalues",
    "gpu_available",
    "inverse_distance_weights",
    "local_moran_panel",
    "memory_diagnostics",
    "moran_panel",
    "select_backend",
    "spatial_lag",
]

__version__ = "0.1.0"
