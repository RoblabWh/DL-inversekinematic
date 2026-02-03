"""Robot visualization and animation module.

This module provides classes for visualizing robots and animating trajectories:
- FigureManager: Centralized figure and animation management
- RobotViewer: Static visualization of robot configurations
- TrajectoryAnimator: Multi-trajectory animation support
- DHExplorer: Interactive DH parameter exploration
- RobotAnimation: Backward-compatible wrapper (deprecated)
"""

import warnings
import copy
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display
from pytransform3d.rotations import plot_basis

from src.robot import Robot


class FigureManager:
    """Centralized manager for matplotlib figures and animations.

    This class-level singleton prevents figure collisions between notebook cells
    by tracking all figures and animations created by the visualization classes.
    """

    _figures: dict[str, plt.Figure] = {}
    _animations: dict[str, animation.FuncAnimation] = {}

    @classmethod
    def get_or_create(cls, name: str, reuse: bool = False) -> tuple[plt.Figure, Axes3D]:
        """Get an existing figure or create a new one.

        Args:
            name: Unique identifier for the figure
            reuse: If True, reuse existing figure with same name; if False, close old and create new

        Returns:
            Tuple of (Figure, Axes3D)
        """
        if name in cls._figures:
            if reuse:
                fig = cls._figures[name]
                # Clear existing axes and create fresh one
                fig.clear()
                ax = fig.add_axes([-0.05, -0.05, 1.20, 1.20], projection='3d', autoscale_on=False)
                return fig, ax
            else:
                cls.close(name)

        fig = plt.figure()
        ax = fig.add_axes([-0.05, -0.05, 1.20, 1.20], projection='3d', autoscale_on=False)
        cls._figures[name] = fig
        return fig, ax

    @classmethod
    def close(cls, name: str) -> None:
        """Close and cleanup a figure by name.

        Args:
            name: Identifier of the figure to close
        """
        if name in cls._animations:
            anim = cls._animations.pop(name)
            anim.event_source.stop()

        if name in cls._figures:
            fig = cls._figures.pop(name)
            plt.close(fig)

    @classmethod
    def close_all(cls) -> None:
        """Close all managed figures and animations."""
        for name in list(cls._figures.keys()):
            cls.close(name)

    @classmethod
    def register_animation(cls, name: str, anim: animation.FuncAnimation) -> None:
        """Register an animation for later cleanup.

        Args:
            name: Figure name this animation belongs to
            anim: The FuncAnimation object
        """
        # Stop any existing animation for this figure
        if name in cls._animations:
            cls._animations[name].event_source.stop()
        cls._animations[name] = anim


class RobotViewer:
    """Static visualization of robot configurations.

    Replaces the old RobotAnimation class for drawing single or multiple
    robot configurations without animation.

    Example:
        viewer = RobotViewer(robot)
        viewer.draw(joint_positions)  # Single configuration
        viewer.draw([config1, config2], colors=["green", "orange"], labels=["GT", "Pred"])
        viewer.close()
    """

    def __init__(self, robot: Robot, show_basis: bool = True, figure_name: Optional[str] = None):
        """Initialize a RobotViewer.

        Args:
            robot: Robot instance to visualize
            show_basis: Whether to show coordinate frames at each joint
            figure_name: Optional custom figure name (defaults to unique per instance)
        """
        self.robot = robot
        self.show_basis = show_basis

        # Compute scale factors based on robot dimensions
        self.scale = np.max([self.robot.dh_values[1], self.robot.dh_values[2]])
        self.robotscale = self.scale * 4587
        self.axscale = np.array([self.robot.dh_values[1], self.robot.dh_values[2]]).sum() * self.robotscale
        self.coordscale = self.axscale / 9

        # Create unique figure name
        self._figure_name = figure_name or f"{self.robot.name}_viewer_{id(self)}"

        # Create figure and axes
        self.fig, self.ax = FigureManager.get_or_create(self._figure_name)
        self.text = self.ax.text(0, 0, self.axscale + self.axscale / 10, s="", va="bottom", ha="left")

        # For manual control
        self._sliders = []
        self._goal = None

    def draw(self, joint_positions, colors=None, labels=None) -> None:
        """Draw one or more robot configurations.

        Args:
            joint_positions: Single array of joint angles (radians) or list of arrays
            colors: Optional color(s) for the robot(s). Defaults to ["orange", "green", "blue"]
            labels: Optional label(s) for legend
        """
        q_arr = np.asarray(joint_positions)

        # Handle single configuration
        if q_arr.ndim == 1:
            q_arr = np.expand_dims(q_arr, 0)

        # Default colors
        default_colors = ["orange", "green", "blue", "red", "purple", "cyan"]
        if colors is None:
            colors = default_colors[:len(q_arr)]
        elif isinstance(colors, str):
            colors = [colors]

        # Extend colors if needed
        while len(colors) < len(q_arr):
            colors.append(default_colors[len(colors) % len(default_colors)])

        # Clear axes for fresh draw
        self.ax.clear()
        self.text = self.ax.text(0, 0, self.axscale + self.axscale / 10, s="", va="bottom", ha="left")

        # Draw each configuration
        for i, q in enumerate(q_arr):
            coordinates, rotation = self.robot.calc_x_y_z(q)
            label = labels[i] if labels and i < len(labels) else None
            self._plot_robot(coordinates, rotation, colors[i], label)

        # Add legend if labels provided
        if labels:
            self.ax.legend(loc='upper right')

        self._set_axes()
        plt.show()

    def _plot_robot(self, coordinates, rotation, color: str, label: Optional[str] = None) -> None:
        """Plot a single robot configuration.

        Args:
            coordinates: XYZ coordinates from robot.calc_x_y_z()
            rotation: Rotation matrices from robot.calc_x_y_z()
            color: Line color
            label: Optional label for legend
        """
        x = (coordinates[0] * self.robotscale).tolist()
        y = (coordinates[1] * self.robotscale).tolist()
        z = (coordinates[2] * self.robotscale).tolist()

        if self.show_basis:
            for i in range(len(self.robot.joint_limits) + 1):
                plot_basis(self.ax, R=rotation[i], p=[x[i], y[i], z[i]], s=self.coordscale)

        self.ax.plot(x, y, z, 'o-', markersize=8,
                     markerfacecolor="black", linewidth=3, color=color, label=label)

    def _set_axes(self) -> None:
        """Configure axis limits and appearance."""
        self.ax.set_xlim3d(-self.axscale, self.axscale)
        self.ax.set_ylim3d(-self.axscale, self.axscale)
        self.ax.set_zlim3d(-self.axscale / 4, self.axscale)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

    def manual_control(self, goal=None) -> None:
        """Enable interactive slider control of robot joints.

        Args:
            goal: Optional goal position [x, y, z] to display
        """
        if goal is not None:
            self._goal = goal * self.robotscale

        self._sliders = []
        for i in range(len(self.robot.joint_limits)):
            min_val = self.robot.joint_limits[i][1]
            max_val = self.robot.joint_limits[i][0]
            slider = widgets.IntSlider(
                value=min_val + max_val,
                min=min_val,
                max=max_val,
                description=f'q{i + 1} [°]'
            )
            slider.observe(self._on_slider_change, names='value')
            self._sliders.append(slider)
            display(slider)

        self._on_slider_change(None)
        plt.show()

    def _on_slider_change(self, change) -> None:
        """Handle slider value changes."""
        q_array = [slider.value for slider in self._sliders]
        q_array = np.around(np.radians(q_array), decimals=4)

        self.ax.clear()
        self.text = self.ax.text(0, 0, self.axscale + self.axscale / 10, s="", va="bottom", ha="left")

        if self._goal is not None:
            self.ax.plot(self._goal[0], self._goal[1], self._goal[2],
                         '-o', markersize=5, markerfacecolor="red")

        coordinates, rotation = self.robot.calc_x_y_z(q_array)
        self._plot_robot(coordinates, rotation, "orange")
        self._set_axes()

    def close(self) -> None:
        """Close this viewer and release resources."""
        FigureManager.close(self._figure_name)


class TrajectoryAnimator:
    """Animate multiple robot trajectories simultaneously.

    Supports displaying multiple trajectories with different colors, labels,
    and optional TCP trails.

    Example:
        animator = TrajectoryAnimator(robot)
        animator.add_trajectory(ground_truth, color="green", label="Ground Truth")
        animator.add_trajectory(predicted, color="orange", label="Prediction")
        animator.animate(show_legend=True)
        animator.close()
    """

    def __init__(self, robot: Robot, show_basis: bool = False, figure_name: Optional[str] = None):
        """Initialize a TrajectoryAnimator.

        Args:
            robot: Robot instance to animate
            show_basis: Whether to show coordinate frames at each joint
            figure_name: Optional custom figure name
        """
        self.robot = robot
        self.show_basis = show_basis

        # Compute scale factors
        self.scale = np.max([self.robot.dh_values[1], self.robot.dh_values[2]])
        self.robotscale = self.scale * 4587
        self.axscale = np.array([self.robot.dh_values[1], self.robot.dh_values[2]]).sum() * self.robotscale
        self.coordscale = self.axscale / 9

        # Create unique figure name
        self._figure_name = figure_name or f"{self.robot.name}_animator_{id(self)}"

        # Create figure and axes
        self.fig, self.ax = FigureManager.get_or_create(self._figure_name)

        # Trajectory storage
        self._trajectories: list[dict] = []
        self._precomputed: list[list] = []  # Precomputed FK for each trajectory

        # Animation state
        self._ani = None
        self._trails: list[dict] = []  # Trail points per trajectory

    def add_trajectory(self, joints: np.ndarray, color: str = "orange",
                       label: Optional[str] = None, show_trail: bool = True) -> None:
        """Add a trajectory to be animated.

        Args:
            joints: Array of shape (N, num_joints) with joint angles in radians
            color: Line color for this trajectory
            label: Optional label for legend
            show_trail: Whether to show TCP trail for this trajectory
        """
        joints = np.asarray(joints)
        if joints.ndim == 1:
            joints = np.expand_dims(joints, 0)

        self._trajectories.append({
            'joints': joints,
            'color': color,
            'label': label,
            'show_trail': show_trail
        })

    def clear_trajectories(self) -> None:
        """Remove all added trajectories."""
        self._trajectories.clear()
        self._precomputed.clear()
        self._trails.clear()

    def animate(self, interval: int = 300, repeat: bool = False, show_legend: bool = True) -> None:
        """Start the animation.

        Args:
            interval: Milliseconds between frames
            repeat: Whether to loop the animation
            show_legend: Whether to display legend
        """
        if not self._trajectories:
            raise ValueError("No trajectories added. Use add_trajectory() first.")

        # Precompute forward kinematics for all trajectories
        self._precompute_fk()

        # Reset trails
        self._trails = [{'x': [], 'y': [], 'z': []} for _ in self._trajectories]

        # Find maximum trajectory length
        max_frames = max(len(traj['joints']) for traj in self._trajectories)

        # Configure axes
        self._set_axes()

        # Create animation with blit=False since we use ax.clear()
        self._ani = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=max_frames,
            interval=interval,
            repeat=repeat,
            blit=False,
            cache_frame_data=False
        )

        FigureManager.register_animation(self._figure_name, self._ani)

        self._show_legend = show_legend
        plt.show()

    def _precompute_fk(self) -> None:
        """Precompute forward kinematics for all trajectories."""
        self._precomputed = []
        for traj in self._trajectories:
            states = []
            for q in traj['joints']:
                coordinates, rotation = self.robot.calc_x_y_z(q)
                states.append((coordinates, rotation))
            self._precomputed.append(states)

    def _update_frame(self, idx: int):
        """Update function for animation."""
        self.ax.clear()
        self._set_axes()

        for i, traj in enumerate(self._trajectories):
            states = self._precomputed[i]

            # Use last frame if this trajectory is shorter
            frame_idx = min(idx, len(states) - 1)
            coordinates, rotation = states[frame_idx]

            # Scale coordinates
            x = coordinates[0] * self.robotscale
            y = coordinates[1] * self.robotscale
            z = coordinates[2] * self.robotscale

            # Update trail (only if trajectory still has frames)
            if traj['show_trail'] and idx <= len(states) - 1:
                self._trails[i]['x'].append(x[-1])
                self._trails[i]['y'].append(y[-1])
                self._trails[i]['z'].append(z[-1])

            # Plot robot
            self._plot_robot(coordinates, rotation, traj['color'], traj['label'])

            # Plot trail
            if traj['show_trail'] and self._trails[i]['x']:
                self.ax.plot_wireframe(
                    np.array([self._trails[i]['x']]),
                    np.array([self._trails[i]['y']]),
                    np.array([self._trails[i]['z']]),
                    color=traj['color'],
                    alpha=0.5
                )

        # Add legend once per frame
        if self._show_legend and any(t['label'] for t in self._trajectories):
            self.ax.legend(loc='upper left')

    def _plot_robot(self, coordinates, rotation, color: str, label: Optional[str] = None) -> None:
        """Plot a single robot configuration."""
        x = (coordinates[0] * self.robotscale).tolist()
        y = (coordinates[1] * self.robotscale).tolist()
        z = (coordinates[2] * self.robotscale).tolist()

        if self.show_basis:
            for i in range(len(self.robot.joint_limits) + 1):
                plot_basis(self.ax, R=rotation[i], p=[x[i], y[i], z[i]], s=self.coordscale)

        self.ax.plot(x, y, z, 'o-', markersize=8,
                     markerfacecolor="black", linewidth=3, color=color, label=label)

    def _set_axes(self) -> None:
        """Configure axis limits and appearance."""
        self.ax.set_xlim3d(-self.axscale, self.axscale)
        self.ax.set_ylim3d(-self.axscale, self.axscale)
        self.ax.set_zlim3d(-self.axscale / 4, self.axscale)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

    def close(self) -> None:
        """Close this animator and release resources."""
        FigureManager.close(self._figure_name)


class DHExplorer:
    """Interactive DH parameter exploration tool.

    Allows users to interactively adjust DH parameters (theta, d, a, alpha)
    to understand how they affect the robot's configuration.

    Example:
        explorer = DHExplorer(robot)
        explorer.start()
    """

    def __init__(self, robot: Robot, figure_name: Optional[str] = None):
        """Initialize a DHExplorer.

        Args:
            robot: Robot instance to explore
            figure_name: Optional custom figure name
        """
        self.robot = robot

        # Compute scale factors
        self.scale = np.max([self.robot.dh_values[1], self.robot.dh_values[2]])
        self.robotscale = self.scale * 4587
        self.axscale = np.array([self.robot.dh_values[1], self.robot.dh_values[2]]).sum() * self.robotscale
        self.coordscale = self.axscale / 9

        # Create unique figure name
        self._figure_name = figure_name or f"{self.robot.name}_dhexplorer_{id(self)}"

        # Create figure and axes
        self.fig, self.ax = FigureManager.get_or_create(self._figure_name)
        self.text = self.ax.text(0, 0, self.axscale + self.axscale / 10, s="", va="bottom", ha="left")

        # DH exploration state
        self.coordinates = None
        self.rotation = None
        self.solved = None
        self.solved_dh_params = None
        self._mutex_transform = False

        # Widget references
        self.angle_slider = None
        self.rot_z = None
        self.trans_z = None
        self.trans_x = None
        self.rot_x = None

        # Current plot references
        self._current_angle = None
        self._current_basis = None

    def start(self, position: Optional[np.ndarray] = None) -> None:
        """Start the DH explorer.

        Args:
            position: Optional initial joint positions (radians)
        """
        if position is None:
            position = np.zeros(len(self.robot.joint_limits))

        # Initialize robot coordinates
        self.robot.calc_x_y_z(position)
        self.coordinates = copy.deepcopy(self.robot.coordinates)
        self.rotation = np.array([np.identity(3) for _ in range(len(self.robot.joint_limits))])
        self.solved = [False] * len(self.robot.joint_limits)
        self.solved_dh_params = np.zeros((len(self.robot.joint_limits) + 1, 4))

        # Draw initial robot (without basis frames for clarity)
        self._draw_initial_robot(position)

        # Create widgets
        self.angle_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=len(self.robot.joint_limits),
            description="Gelenk"
        )
        self.angle_slider.observe(self._on_angle_slider_change, names='value')
        display(self.angle_slider)

        self.rot_z = widgets.BoundedFloatText(
            value=0, min=-180, max=180, step=1,
            description="Rot θ [°]"
        )
        self.rot_z.observe(self._rotate_theta, names='value')
        display(self.rot_z)

        max_trans = np.array(self.robot.dh_values[2:]).max()

        self.trans_z = widgets.BoundedFloatText(
            value=0, min=0, max=max_trans, step=0.01,
            description="Trans d [m]"
        )
        self.trans_z.observe(self._translate_z, names='value')
        display(self.trans_z)

        self.trans_x = widgets.BoundedFloatText(
            value=0, min=0, max=max_trans, step=0.01,
            description="Trans a [m]"
        )
        self.trans_x.observe(self._translate_x, names='value')
        display(self.trans_x)

        self.rot_x = widgets.BoundedFloatText(
            value=0, min=-180, max=180, step=1,
            description="Rot α [°]"
        )
        self.rot_x.observe(self._rotate_alpha, names='value')
        display(self.rot_x)

        # Trigger initial update
        self._on_angle_slider_change({"new": 1, "old": 1})
        plt.show()

    def _draw_initial_robot(self, position: np.ndarray) -> None:
        """Draw the initial robot configuration without basis frames."""
        coordinates, rotation = self.robot.calc_x_y_z(position)
        x = (coordinates[0] * self.robotscale).tolist()
        y = (coordinates[1] * self.robotscale).tolist()
        z = (coordinates[2] * self.robotscale).tolist()

        self.ax.plot(x, y, z, 'o-', markersize=8,
                     markerfacecolor="black", linewidth=3, color="blue", alpha=0.5)
        self._set_axes()

    def _set_axes(self) -> None:
        """Configure axis limits and appearance."""
        self.ax.set_xlim3d(-self.axscale, self.axscale)
        self.ax.set_ylim3d(-self.axscale, self.axscale)
        self.ax.set_zlim3d(-self.axscale / 4, self.axscale)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

    def _check_ground_truth(self) -> None:
        """Check if current configuration matches ground truth."""
        i = self.angle_slider.value - 1
        a = self.robot.rotation[i + 1]
        b = self.rotation[i]
        xyz_gt = np.array([
            self.robot.coordinates[0][i + 1],
            self.robot.coordinates[1][i + 1],
            self.robot.coordinates[2][i + 1]
        ])
        xyz = np.array([
            self.coordinates[0][i],
            self.coordinates[1][i],
            self.coordinates[2][i]
        ])

        if np.allclose(a, b, rtol=1e-10, atol=1e-03) and np.allclose(xyz, xyz_gt, rtol=1e-10, atol=1e-03):
            if all(self.solved[:i]):
                self.solved[i] = True
                self.text.set_text("Dieser Winkel ist korrekt.\nWeiter zum Nächsten!")
                self.solved_dh_params[i] = [
                    self.rot_z.value, self.trans_z.value,
                    self.trans_x.value, self.rot_x.value
                ]
        else:
            self.solved[i] = False
            self.solved_dh_params[i] = [0, 0, 0, 0]
            self.text.set_text("")

    def _translate_z(self, change) -> None:
        """Handle d (z translation) changes."""
        if not self._mutex_transform:
            self._clear_current_plot()
            translation_z_d = self.trans_z.value - change['old']
            angle = self.angle_slider.value - 1

            trans_d = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, translation_z_d],
                [0, 0, 0, 1]
            ])

            robot_rotation = self.rotation[angle].tolist()
            for i in range(3):
                robot_rotation[i].append(self.coordinates[i][angle])
            robot_rotation.append([0, 0, 0, 1])
            robot_rotation = np.array(robot_rotation)

            dh = robot_rotation.dot(trans_d)
            xyz = dh[:-1, -1]
            self.coordinates[0][angle] = xyz[0]
            self.coordinates[1][angle] = xyz[1]
            self.coordinates[2][angle] = xyz[2]

            self._plot_current_coords(angle)

    def _translate_x(self, change) -> None:
        """Handle a (x translation) changes."""
        if not self._mutex_transform:
            self._clear_current_plot()
            translation_x_a = self.trans_x.value - change['old']
            angle = self.angle_slider.value - 1

            trans_a = np.array([
                [1, 0, 0, translation_x_a],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            robot_rotation = self.rotation[angle].tolist()
            for i in range(3):
                robot_rotation[i].append(self.coordinates[i][angle])
            robot_rotation.append([0, 0, 0, 1])
            robot_rotation = np.array(robot_rotation)

            dh = robot_rotation.dot(trans_a)
            xyz = dh[:-1, -1]
            self.coordinates[0][angle] = xyz[0]
            self.coordinates[1][angle] = xyz[1]
            self.coordinates[2][angle] = xyz[2]

            self._plot_current_coords(angle)

    def _rotate_theta(self, change) -> None:
        """Handle theta (z rotation) changes."""
        if not self._mutex_transform:
            self._clear_current_plot()
            rotation_angle_theta = np.deg2rad(self.rot_z.value - change['old'])

            rotation_theta = np.array([
                [np.cos(rotation_angle_theta), -np.sin(rotation_angle_theta), 0],
                [np.sin(rotation_angle_theta), np.cos(rotation_angle_theta), 0],
                [0, 0, 1]
            ])

            angle = self.angle_slider.value - 1
            robot_rotation = np.asarray(self.rotation[angle])
            self.rotation[angle] = robot_rotation.dot(rotation_theta)

            self._plot_current_coords(angle)

    def _rotate_alpha(self, change) -> None:
        """Handle alpha (x rotation) changes."""
        if not self._mutex_transform:
            self._clear_current_plot()
            rotation_angle_alpha = np.deg2rad(self.rot_x.value - change['old'])

            rotation_alpha = np.array([
                [1, 0, 0],
                [0, np.cos(rotation_angle_alpha), -np.sin(rotation_angle_alpha)],
                [0, np.sin(rotation_angle_alpha), np.cos(rotation_angle_alpha)]
            ])

            angle = self.angle_slider.value - 1
            robot_rotation = np.asarray(self.rotation[angle])
            self.rotation[angle] = robot_rotation.dot(rotation_alpha)

            self._plot_current_coords(angle)

    def _on_angle_slider_change(self, change) -> None:
        """Handle joint selection changes."""
        angle = change['new'] - 1
        old_angle = change['old'] - 1

        self._mutex_transform = True
        params = self.solved_dh_params[angle]
        self.rot_z.value = params[0]
        self.trans_z.value = params[1]
        self.trans_x.value = params[2]
        self.rot_x.value = params[3]
        self._mutex_transform = False

        if not self.solved[old_angle]:
            self.rotation[old_angle] = np.identity(3)
            self.coordinates[0][old_angle] = copy.deepcopy(self.robot.coordinates[0][old_angle])
            self.coordinates[1][old_angle] = copy.deepcopy(self.robot.coordinates[1][old_angle])
            self.coordinates[2][old_angle] = copy.deepcopy(self.robot.coordinates[2][old_angle])

        if angle != 0 and self.solved[angle - 1] and not self.solved[angle]:
            self.rotation[angle] = copy.deepcopy(self.robot.rotation[angle])
            self.coordinates[0][angle] = copy.deepcopy(self.robot.coordinates[0][angle])
            self.coordinates[1][angle] = copy.deepcopy(self.robot.coordinates[1][angle])
            self.coordinates[2][angle] = copy.deepcopy(self.robot.coordinates[2][angle])

        self._clear_current_plot()
        self._plot_current_coords(angle)

    def _plot_current_coords(self, angle: int) -> None:
        """Plot the current coordinate frame being edited."""
        x = (self.coordinates[0][angle] * self.robotscale).tolist()
        y = (self.coordinates[1][angle] * self.robotscale).tolist()
        z = (self.coordinates[2][angle] * self.robotscale).tolist()

        self._current_basis = plot_basis(
            self.ax, R=self.rotation[angle],
            p=[x, y, z], s=self.coordscale
        )
        self._current_angle = self.ax.plot(
            x, y, z, 'o-', markersize=3,
            markerfacecolor="orange", linewidth=3, color="orange"
        )

        self._check_ground_truth()

    def _clear_current_plot(self) -> None:
        """Remove the current coordinate frame plot."""
        if self._current_angle:
            self._current_angle[0].remove()
            self._current_angle = None
        if self._current_basis:
            for i, line in enumerate(self._current_basis.get_lines()):
                if i != 0:
                    line.remove()
            self._current_basis = None

    def close(self) -> None:
        """Close this explorer and release resources."""
        FigureManager.close(self._figure_name)


class RobotAnimation:
    """Backward-compatible wrapper for legacy code.

    .. deprecated::
        Use RobotViewer for static visualization or TrajectoryAnimator
        for animations instead.
    """

    def __init__(self, robot: Robot, basis: bool = True):
        """Initialize a legacy RobotAnimation wrapper.

        Args:
            robot: Robot instance
            basis: Whether to show coordinate frames
        """
        warnings.warn(
            "RobotAnimation is deprecated. Use RobotViewer for static visualization "
            "or TrajectoryAnimator for animations.",
            DeprecationWarning,
            stacklevel=2
        )

        self.robot = robot
        self._viewer = RobotViewer(robot, show_basis=basis)
        self._animator = None
        self._explorer = None

        # Expose some properties for compatibility
        self.fig = self._viewer.fig
        self.ax = self._viewer.ax
        self.scale = self._viewer.scale
        self.robotscale = self._viewer.robotscale
        self.axscale = self._viewer.axscale
        self.coordscale = self._viewer.coordscale

    def draw_robot(self, q_array) -> None:
        """Draw robot configuration(s).

        Args:
            q_array: Joint positions (single or multiple configurations)
        """
        self._viewer.draw(q_array)

    def draw_trajectory_robot(self, trajectory, repeat: bool = False) -> None:
        """Animate a robot trajectory.

        Args:
            trajectory: Array of joint configurations
            repeat: Whether to loop the animation
        """
        # Close viewer figure before creating animator
        self._viewer.close()

        self._animator = TrajectoryAnimator(self.robot, show_basis=False)
        self._animator.add_trajectory(trajectory, color="orange", show_trail=True)
        self._animator.animate(interval=300, repeat=repeat, show_legend=False)

        # Update fig/ax references
        self.fig = self._animator.fig
        self.ax = self._animator.ax

    def manual_control(self, goal=None) -> None:
        """Enable interactive slider control.

        Args:
            goal: Optional goal position
        """
        self._viewer.manual_control(goal)

    def dh_animation(self, position=None) -> None:
        """Start DH parameter exploration.

        Args:
            position: Optional initial joint positions
        """
        # Close viewer figure before creating explorer
        self._viewer.close()

        self._explorer = DHExplorer(self.robot)
        self._explorer.start(position)

        # Update fig/ax references
        self.fig = self._explorer.fig
        self.ax = self._explorer.ax


# Alias for backward compatibility
DhAnimation = DHExplorer
