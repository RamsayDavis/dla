"""
DLA Simulation Library - Production Core Models

This package provides three core DLA simulators:
- LatticeSimulator: On-lattice DLA with hierarchical acceleration
- BellOffSimulator: Off-lattice DLA with conformal mapping
- HybridSimulator: Hybrid off-lattice diffusion with on-lattice aggregation
"""

from .ongrid_sim import LatticeConfig, LatticeSimulator
from .offgrid_sim import BellOffParams, BellOffSimulator
from .hybrid_sim import HybridParams, HybridSimulator
from . import utils

__all__ = [
    # Simulators
    "LatticeSimulator",
    "BellOffSimulator", 
    "HybridSimulator",
    # Configuration classes
    "LatticeConfig",
    "BellOffParams",
    "HybridParams",
    # Utilities
    "utils",
]

