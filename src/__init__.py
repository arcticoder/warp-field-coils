"""
Warp Field Coils - Advanced Propulsion System
===========================================

A comprehensive implementation of steerable warp field technology with
full 3D control, numerical stability, and integrated LQG corrections.
"""

__version__ = "1.0.0"
__author__ = "Warp Drive Research Team"

# Lazy imports to avoid loading heavy dependencies at module level
# Users should import specific modules directly when needed

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
