"""Shared metrics module for IK evaluation.

Provides joint-space and TCP-space error metrics used by both
the training script (src/train.py) and Notebook 2.
"""

import numpy as np
from src.robot import Robot


def compute_joint_metrics(gt_joints: np.ndarray, pred_joints: np.ndarray) -> dict:
    """Joint-space error metrics.

    Args:
        gt_joints: Ground truth joint angles (N, num_joints) in radians.
        pred_joints: Predicted joint angles (N, num_joints) in radians.

    Returns:
        Dictionary with MAE, RMSE, max error, percentiles, and success rates.
    """
    errors = np.abs(gt_joints - pred_joints)
    errors_deg = np.degrees(errors)

    mae_per_joint_rad = np.mean(errors, axis=0)
    mae_per_joint_deg = np.degrees(mae_per_joint_rad)
    mae_total_rad = float(np.mean(mae_per_joint_rad))
    mae_total_deg = float(np.degrees(mae_total_rad))

    rmse_total_rad = float(np.sqrt(np.mean((gt_joints - pred_joints) ** 2)))
    rmse_total_deg = float(np.degrees(rmse_total_rad))

    max_error_per_joint_deg = np.max(errors_deg, axis=0)

    p50_per_joint_deg = np.percentile(errors_deg, 50, axis=0)
    p95_per_joint_deg = np.percentile(errors_deg, 95, axis=0)
    p99_per_joint_deg = np.percentile(errors_deg, 99, axis=0)

    # Success rate: fraction of samples where ALL joints are below threshold
    max_error_per_sample_deg = np.max(errors_deg, axis=1)
    success_rate_1deg = float(np.mean(max_error_per_sample_deg < 1.0))
    success_rate_5deg = float(np.mean(max_error_per_sample_deg < 5.0))

    return {
        "mae_per_joint_rad": mae_per_joint_rad,
        "mae_per_joint_deg": mae_per_joint_deg,
        "mae_total_rad": mae_total_rad,
        "mae_total_deg": mae_total_deg,
        "rmse_total_rad": rmse_total_rad,
        "rmse_total_deg": rmse_total_deg,
        "max_error_per_joint_deg": max_error_per_joint_deg,
        "p50_per_joint_deg": p50_per_joint_deg,
        "p95_per_joint_deg": p95_per_joint_deg,
        "p99_per_joint_deg": p99_per_joint_deg,
        "success_rate_1deg": success_rate_1deg,
        "success_rate_5deg": success_rate_5deg,
    }


def compute_tcp_metrics(gt_joints: np.ndarray, pred_joints: np.ndarray, robot: Robot) -> dict:
    """TCP-space error via forward kinematics.

    Runs FK on both ground truth and predicted joints, then compares
    end-effector position (Euclidean) and orientation (geodesic angle).

    Args:
        gt_joints: Ground truth joint angles (N, num_joints) in radians.
        pred_joints: Predicted joint angles (N, num_joints) in radians.
        robot: Robot instance (will be temporarily set to numpy mode).

    Returns:
        Dictionary with position and orientation error statistics.
    """
    # Save and restore torch state
    was_torch = robot.use_torch
    saved_device = robot.device
    if was_torch:
        robot.set_torch(False)

    try:
        gt_frames = robot.buildDhTcpFrame(gt_joints)
        pred_frames = robot.buildDhTcpFrame(pred_joints)
    finally:
        if was_torch:
            robot.set_torch(True, device=saved_device)

    # Position error (Euclidean distance between TCP positions)
    gt_pos = gt_frames[:, :3, 3]
    pred_pos = pred_frames[:, :3, 3]
    pos_errors = np.linalg.norm(gt_pos - pred_pos, axis=1)

    # Orientation error (geodesic angle between rotation matrices)
    gt_rot = gt_frames[:, :3, :3]
    pred_rot = pred_frames[:, :3, :3]
    # R_diff = R_gt^T @ R_pred
    r_diff = np.einsum("nij,njk->nik", np.transpose(gt_rot, (0, 2, 1)), pred_rot)
    trace_vals = np.trace(r_diff, axis1=1, axis2=2)
    # Clamp for numerical stability: arccos domain is [-1, 1]
    cos_angle = np.clip((trace_vals - 1.0) / 2.0, -1.0, 1.0)
    orient_errors_rad = np.arccos(cos_angle)
    orient_errors_deg = np.degrees(orient_errors_rad)

    # Success rates (position thresholds in meters)
    success_rate_1mm = float(np.mean(pos_errors < 0.001))
    success_rate_5mm = float(np.mean(pos_errors < 0.005))

    return {
        "position_error_mean": float(np.mean(pos_errors)),
        "position_error_median": float(np.median(pos_errors)),
        "position_error_max": float(np.max(pos_errors)),
        "position_error_p95": float(np.percentile(pos_errors, 95)),
        "position_error_p99": float(np.percentile(pos_errors, 99)),
        "orientation_error_mean_deg": float(np.mean(orient_errors_deg)),
        "orientation_error_max_deg": float(np.max(orient_errors_deg)),
        "success_rate_1mm": success_rate_1mm,
        "success_rate_5mm": success_rate_5mm,
    }


def compute_all_metrics(gt_joints: np.ndarray, pred_joints: np.ndarray, robot: Robot = None) -> dict:
    """Compute joint-space and optionally TCP-space metrics.

    Args:
        gt_joints: Ground truth joint angles (N, num_joints) in radians.
        pred_joints: Predicted joint angles (N, num_joints) in radians.
        robot: If provided, also computes TCP-space metrics via FK.

    Returns:
        {"joint": {...}, "tcp": {...} or None}
    """
    result = {
        "joint": compute_joint_metrics(gt_joints, pred_joints),
        "tcp": None,
    }
    if robot is not None:
        result["tcp"] = compute_tcp_metrics(gt_joints, pred_joints, robot)
    return result


def format_metrics(metrics: dict) -> str:
    """Format metrics dictionary as a readable table.

    Position errors are displayed in millimeters for readability.

    Args:
        metrics: Output from compute_all_metrics().

    Returns:
        Formatted string.
    """
    lines = []
    jm = metrics["joint"]
    num_joints = len(jm["mae_per_joint_deg"])

    # Header
    lines.append("--- Joint-Space Metrics ---")
    lines.append(f"MAE Total:  {jm['mae_total_rad']:.6f} rad  ({jm['mae_total_deg']:.2f} deg)")
    lines.append(f"RMSE Total: {jm['rmse_total_rad']:.6f} rad  ({jm['rmse_total_deg']:.2f} deg)")
    lines.append(f"Success Rate: <1deg {jm['success_rate_1deg']:.1%}  |  <5deg {jm['success_rate_5deg']:.1%}")
    lines.append("")

    # Per-joint table
    header = f"{'Joint':<7} {'MAE(deg)':>9} {'Max(deg)':>9} {'P50(deg)':>9} {'P95(deg)':>9} {'P99(deg)':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for i in range(num_joints):
        lines.append(
            f"  {i+1:<5} {jm['mae_per_joint_deg'][i]:>9.2f} "
            f"{jm['max_error_per_joint_deg'][i]:>9.2f} "
            f"{jm['p50_per_joint_deg'][i]:>9.2f} "
            f"{jm['p95_per_joint_deg'][i]:>9.2f} "
            f"{jm['p99_per_joint_deg'][i]:>9.2f}"
        )

    tm = metrics.get("tcp")
    if tm is not None:
        lines.append("")
        lines.append("--- TCP-Space Metrics ---")
        lines.append(
            f"Position Error:  mean {tm['position_error_mean']*1000:.2f} mm  |  "
            f"median {tm['position_error_median']*1000:.2f} mm  |  "
            f"max {tm['position_error_max']*1000:.2f} mm"
        )
        lines.append(
            f"                 P95 {tm['position_error_p95']*1000:.2f} mm  |  "
            f"P99 {tm['position_error_p99']*1000:.2f} mm"
        )
        lines.append(
            f"Orientation Error: mean {tm['orientation_error_mean_deg']:.2f} deg  |  "
            f"max {tm['orientation_error_max_deg']:.2f} deg"
        )
        lines.append(
            f"Success Rate: <1mm {tm['success_rate_1mm']:.1%}  |  <5mm {tm['success_rate_5mm']:.1%}"
        )

    return "\n".join(lines)
