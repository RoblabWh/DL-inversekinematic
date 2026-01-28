"""Trajectory generators for IK visualization."""
import numpy as np
from typing import List, Dict, Any, Callable

# Registry of trajectory generators
_TRAJECTORIES: Dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a trajectory generator."""
    def decorator(fn):
        _TRAJECTORIES[name] = fn
        return fn
    return decorator


def list_trajectories() -> List[str]:
    """Return list of available trajectory names."""
    return list(_TRAJECTORIES.keys())


def get_trajectory(name: str, robot, n_frames: int = 60, **kwargs) -> np.ndarray:
    """Generate a trajectory for the given robot.

    Args:
        name: Trajectory type (see list_trajectories())
        robot: Robot instance (provides joint_limits and num_joints)
        n_frames: Number of frames to generate
        **kwargs: Trajectory-specific parameters

    Returns:
        numpy array of shape (n_frames, num_joints) in radians
    """
    if name not in _TRAJECTORIES:
        raise ValueError(f"Unknown trajectory: {name}. Available: {list_trajectories()}")

    # Extract joint limits in radians
    limits_deg = np.array(robot.joint_limits)  # Shape: (num_joints, 2) as [max, min]
    limits_rad = np.radians(limits_deg)

    return _TRAJECTORIES[name](limits_rad, n_frames, **kwargs)


# --- Trajectory Generators ---

@register("circle")
def circle(limits: np.ndarray, n_frames: int, **kwargs) -> np.ndarray:
    """Sweep joint 1 from max to min while others stay at midpoint."""
    num_joints = limits.shape[0]
    midpoints = (limits[:, 0] + limits[:, 1]) / 2

    trajectory = np.tile(midpoints, (n_frames, 1))
    trajectory[:, 0] = np.linspace(limits[0, 0], limits[0, 1], n_frames)
    return trajectory


@register("wave")
def wave(limits: np.ndarray, n_frames: int, amplitude: float = 0.7, **kwargs) -> np.ndarray:
    """Sinusoidal wave on all joints with phase offset."""
    num_joints = limits.shape[0]
    midpoints = (limits[:, 0] + limits[:, 1]) / 2
    ranges = (limits[:, 0] - limits[:, 1]) / 2 * amplitude

    t = np.linspace(0, 2 * np.pi, n_frames)
    trajectory = np.zeros((n_frames, num_joints))

    for j in range(num_joints):
        phase = j * np.pi / num_joints  # Phase offset per joint
        trajectory[:, j] = midpoints[j] + ranges[j] * np.sin(t + phase)

    return trajectory


@register("reach")
def reach(limits: np.ndarray, n_frames: int, **kwargs) -> np.ndarray:
    """Extend arm outward then retract (affects elbow/wrist joints)."""
    num_joints = limits.shape[0]
    midpoints = (limits[:, 0] + limits[:, 1]) / 2

    trajectory = np.tile(midpoints, (n_frames, 1))

    # Create reach motion: out for first half, back for second half
    half = n_frames // 2

    if num_joints >= 2:
        # Joint 1 (shoulder/elbow): go from mid to extended
        trajectory[:half, 1] = np.linspace(midpoints[1], limits[1, 0] * 0.8, half)
        trajectory[half:, 1] = np.linspace(limits[1, 0] * 0.8, midpoints[1], n_frames - half)

    if num_joints >= 3:
        # Joint 2 (elbow/wrist): complementary motion
        trajectory[:half, 2] = np.linspace(midpoints[2], limits[2, 1] * 0.8, half)
        trajectory[half:, 2] = np.linspace(limits[2, 1] * 0.8, midpoints[2], n_frames - half)

    return trajectory


@register("spiral")
def spiral(limits: np.ndarray, n_frames: int, rotations: float = 2.0, **kwargs) -> np.ndarray:
    """Spiral: rotate joint 1 while reaching with others."""
    num_joints = limits.shape[0]
    midpoints = (limits[:, 0] + limits[:, 1]) / 2

    trajectory = np.tile(midpoints, (n_frames, 1))

    t = np.linspace(0, rotations * 2 * np.pi, n_frames)

    # Joint 0: full rotation sweep
    j0_range = (limits[0, 0] - limits[0, 1]) / 2 * 0.9
    trajectory[:, 0] = midpoints[0] + j0_range * np.sin(t)

    # Other joints: expanding/contracting reach
    reach_profile = 0.5 + 0.3 * np.sin(t / rotations)
    for j in range(1, min(3, num_joints)):
        j_range = (limits[j, 0] - limits[j, 1]) / 2
        trajectory[:, j] = midpoints[j] + j_range * reach_profile * (0.7 if j == 1 else -0.5)

    return trajectory


@register("random_smooth")
def random_smooth(limits: np.ndarray, n_frames: int, n_waypoints: int = 5,
                  seed: int = None, **kwargs) -> np.ndarray:
    """Random waypoints with cubic spline interpolation."""
    from scipy.interpolate import CubicSpline

    if seed is not None:
        np.random.seed(seed)

    num_joints = limits.shape[0]

    # Generate random waypoints within 80% of joint limits
    margin = 0.1
    low = limits[:, 1] + margin * (limits[:, 0] - limits[:, 1])
    high = limits[:, 0] - margin * (limits[:, 0] - limits[:, 1])

    waypoint_times = np.linspace(0, 1, n_waypoints)
    waypoints = np.random.uniform(low, high, (n_waypoints, num_joints))

    # Make it loop back smoothly
    waypoints[-1] = waypoints[0]

    # Interpolate each joint
    t_fine = np.linspace(0, 1, n_frames)
    trajectory = np.zeros((n_frames, num_joints))

    for j in range(num_joints):
        cs = CubicSpline(waypoint_times, waypoints[:, j], bc_type='periodic')
        trajectory[:, j] = cs(t_fine)

    return trajectory


@register("figure_eight")
def figure_eight(limits: np.ndarray, n_frames: int, **kwargs) -> np.ndarray:
    """Figure-8 pattern using joints 0 and 1."""
    num_joints = limits.shape[0]
    midpoints = (limits[:, 0] + limits[:, 1]) / 2

    trajectory = np.tile(midpoints, (n_frames, 1))

    t = np.linspace(0, 2 * np.pi, n_frames)

    # Lissajous curve parameters for figure-8
    j0_range = (limits[0, 0] - limits[0, 1]) / 2 * 0.7
    trajectory[:, 0] = midpoints[0] + j0_range * np.sin(t)

    if num_joints >= 2:
        j1_range = (limits[1, 0] - limits[1, 1]) / 2 * 0.5
        trajectory[:, 1] = midpoints[1] + j1_range * np.sin(2 * t)

    return trajectory


@register("square")
def square(limits: np.ndarray, n_frames: int, **kwargs) -> np.ndarray:
    """Square pattern using joints 0 and 1 with sharp corners."""
    num_joints = limits.shape[0]
    midpoints = (limits[:, 0] + limits[:, 1]) / 2
    trajectory = np.tile(midpoints, (n_frames, 1))

    j0_range = (limits[0, 0] - limits[0, 1]) / 2 * 0.6
    j1_range = (limits[1, 0] - limits[1, 1]) / 2 * 0.5 if num_joints >= 2 else 0

    # 4 segments: right, up, left, down
    for i in range(n_frames):
        phase = i / n_frames
        if phase < 0.25:      # right
            trajectory[i, 0] = midpoints[0] + j0_range * (phase * 4)
        elif phase < 0.5:     # up
            trajectory[i, 0] = midpoints[0] + j0_range
            if num_joints >= 2:
                trajectory[i, 1] = midpoints[1] + j1_range * ((phase - 0.25) * 4)
        elif phase < 0.75:    # left
            trajectory[i, 0] = midpoints[0] + j0_range * (1 - (phase - 0.5) * 4)
            if num_joints >= 2:
                trajectory[i, 1] = midpoints[1] + j1_range
        else:                 # down
            trajectory[i, 0] = midpoints[0]
            if num_joints >= 2:
                trajectory[i, 1] = midpoints[1] + j1_range * (1 - (phase - 0.75) * 4)
    return trajectory


@register("pringles")
def pringles(limits: np.ndarray, n_frames: int, **kwargs) -> np.ndarray:
    """True pringles in joint space using joints 0 and 1 with 90° phase."""
    num_joints = limits.shape[0]
    midpoints = (limits[:, 0] + limits[:, 1]) / 2
    trajectory = np.tile(midpoints, (n_frames, 1))

    t = np.linspace(0, 2 * np.pi, n_frames)
    j0_range = (limits[0, 0] - limits[0, 1]) / 2 * 0.6
    trajectory[:, 0] = midpoints[0] + j0_range * np.cos(t)

    if num_joints >= 2:
        j1_range = (limits[1, 0] - limits[1, 1]) / 2 * 0.5
        trajectory[:, 1] = midpoints[1] + j1_range * np.sin(t)
    return trajectory
