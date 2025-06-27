"""
Control systems module for closed-loop field control and dynamic trajectory control
"""

# Import order matters to avoid circular imports
try:
    from .dynamic_trajectory_controller import DynamicTrajectoryController, TrajectoryParams, TrajectoryState
    DYNAMIC_AVAILABLE = True
except ImportError as e:
    print(f"Dynamic trajectory controller not available: {e}")
    DYNAMIC_AVAILABLE = False

try:
    from .multi_axis_controller import MultiAxisController, MultiAxisParams
    MULTI_AXIS_AVAILABLE = True
except ImportError as e:
    print(f"Multi-axis controller not available: {e}")
    MULTI_AXIS_AVAILABLE = False

try:
    from .closed_loop_controller import ClosedLoopFieldController, PlantParams, ControllerParams, ControlPerformance, TransferFunction
    CLOSED_LOOP_AVAILABLE = True
except ImportError as e:
    print(f"Closed loop controller not available: {e}")
    CLOSED_LOOP_AVAILABLE = False

# Build __all__ list dynamically
__all__ = []

if CLOSED_LOOP_AVAILABLE:
    __all__.extend(['ClosedLoopFieldController', 'PlantParams', 'ControllerParams', 'ControlPerformance', 'TransferFunction'])

if DYNAMIC_AVAILABLE:
    __all__.extend(['DynamicTrajectoryController', 'TrajectoryParams', 'TrajectoryState'])

if MULTI_AXIS_AVAILABLE:
    __all__.extend(['MultiAxisController', 'MultiAxisParams'])
