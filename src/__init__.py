"""
Warp Field Coils - Advanced Propulsion System
===========================================

A comprehensive implementation of steerable warp field technology with
full 3D control, numerical stability, and integrated LQG corrections.
"""

__version__ = "1.0.0"
__author__ = "Warp Drive Research Team"

# Core imports
from .field_solver import ElectromagneticFieldSolver
from .coil_optimizer import CoilGeometryOptimizer, CurrentDistributionOptimizer
from .integration import NegativeEnergyInterface, WarpBubbleInterface
from .hardware import FieldActuator, CurrentDriver

# Advanced control systems
from .control import (
    MultiAxisController,
    DynamicTrajectoryController,
    TransferFunction
)

# Communication systems
from .subspace_transceiver import (
    SubspaceTransceiver,
    SubspaceParams,
    TransmissionParams
)

# Holographic force-field systems
from .holodeck_forcefield_grid import (
    ForceFieldGrid,
    GridParams,
    Node
)

# Medical tractor arrays
from .medical_tractor_array import (
    MedicalTractorArray,
    MedicalArrayParams,
    TractorBeam,
    BeamMode,
    SafetyLevel,
    VitalSigns
)

__all__ = [
    # Core systems
    "ElectromagneticFieldSolver",
    "CoilGeometryOptimizer", 
    "CurrentDistributionOptimizer",
    "NegativeEnergyInterface",
    "WarpBubbleInterface",
    "FieldActuator",
    "CurrentDriver",
    
    # Advanced control
    "MultiAxisController",
    "DynamicTrajectoryController", 
    "TransferFunction",
    
    # Communication
    "SubspaceTransceiver",
    "SubspaceParams",
    "TransmissionParams",
    
    # Holodeck systems
    "ForceFieldGrid",
    "GridParams", 
    "Node",
    
    # Medical systems
    "MedicalTractorArray",
    "MedicalArrayParams",
    "TractorBeam",
    "BeamMode",
    "SafetyLevel",
    "VitalSigns"
]
