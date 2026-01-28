# DL-Inverse Kinematics

> Deep learning approach to solving inverse kinematics for robot manipulators

An educational project that teaches deep learning concepts by solving inverse kinematics for robot arms using PyTorch. Learn how to build and train neural networks while exploring robotics fundamentals.

## Features

- PyTorch-based neural network for IK solving
- Support for multiple robot configurations (2-7 axis)
- DH (Denavit-Hartenberg) parameter transformations
- Dual NumPy/PyTorch support for flexibility
- 3D visualization with matplotlib
- Hyperparameter optimization with Optuna
- Interactive Jupyter notebooks for learning

## Installation

### Using uv (Recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

## Usage

### Jupyter Notebooks

The project includes interactive notebooks for learning:

1. **1-Robotsimulation-and-DH-Transformation.ipynb** - Introduction to DH transformations and robot simulation
2. **2-Deep Learning Inverse Kinematic.ipynb** - Training neural networks for inverse kinematics
3. **3-Inverse Kinematic-SOM.ipynb** - Self-organizing maps approach

Start Jupyter:
```bash
# Using uv
uv run jupyter notebook

# Or if using pip installation
jupyter notebook
```

## Project Structure

```
DL-inversekinematic/
├── src/
│   ├── robot.py          # Robot class with DH transformations
│   ├── datahandler.py    # Data generation and normalization
│   ├── trainer.py        # PyTorch training loop
│   ├── robot_animation.py # 3D visualization
│   ├── robot_plot.py     # Static robot plotting
│   └── display_latex.py  # LaTeX rendering utilities
├── 1-Robotsimulation-and-DH-Transformation.ipynb
├── 2-Deep Learning Inverse Kinematic.ipynb
├── 3-Inverse Kinematic-SOM.ipynb
├── pyproject.toml
└── README.md
```

## Supported Robots

| Robot | Axes | Description |
|-------|------|-------------|
| `youbot` | 5 | KUKA YouBot manipulator arm |
| `baxter` | 7 | Rethink Robotics Baxter arm |
| `twoaxis` | 2 | Generic 2-axis planar manipulator |
| `threeaxis` | 3 | Generic 3-axis manipulator |
| `fouraxis` | 4 | Generic 4-axis manipulator |
| `fiveaxis` | 5 | Generic 5-axis manipulator |

Usage:
```python
from src.robot import Robot
robot = Robot(robot='youbot')
```

## References

- [narcispr/dnn_inverse_kinematics](https://github.com/narcispr/dnn_inverse_kinematics) - Iterative IK calculation with Jacobian method
