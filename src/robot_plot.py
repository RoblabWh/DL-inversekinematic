"""Deprecated robot plotting module.

This module is deprecated. The original functionality has been moved to
robot_plot_deprecated.py for backward compatibility.
"""

import warnings

warnings.warn(
    "robot_plot is deprecated. Use RobotViewer from robot_animation instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.robot_plot_deprecated import robot_pose, robot_pose2

__all__ = ['robot_pose', 'robot_pose2']
