"""
Warp Field Coils - Core Package

This package provides electromagnetic field optimization and coil geometry design
for warp drive propulsion systems, integrating with negative energy generation.
"""

__version__ = "0.1.0"
__author__ = "Warp Drive Research Team"

# Core imports
from .field_solver import ElectromagneticFieldSolver
from .coil_optimizer import CoilGeometryOptimizer, CurrentDistributionOptimizer
from .integration import NegativeEnergyInterface, WarpBubbleInterface
from .hardware import FieldActuator, CurrentDriver

__all__ = [
    "ElectromagneticFieldSolver",
    "CoilGeometryOptimizer", 
    "CurrentDistributionOptimizer",
    "NegativeEnergyInterface",
    "WarpBubbleInterface",
    "FieldActuator",
    "CurrentDriver"
]
