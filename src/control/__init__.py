"""
Control systems module for closed-loop field control and dynamic trajectory control
"""

from .closed_loop_controller import ClosedLoopFieldController, PlantParams, ControllerParams, ControlPerformance

try:
    from .dynamic_trajectory_controller import DynamicTrajectoryController, TrajectoryParams, TrajectoryState
    __all__ = ['ClosedLoopFieldController', 'PlantParams', 'ControllerParams', 'ControlPerformance',
               'DynamicTrajectoryController', 'TrajectoryParams', 'TrajectoryState']
except ImportError:
    __all__ = ['ClosedLoopFieldController', 'PlantParams', 'ControllerParams', 'ControlPerformance']
