# CLAUDE.md

## Quick Reference

- **Language**: Python 3
- **Framework**: PyTorch (with NumPy fallback)
- **Purpose**: Educational IK solver using deep learning
- **Entry points**: Jupyter notebooks

## Architecture Overview

### Core Classes

| Class | File | Purpose |
|-------|------|---------|
| `Robot` | `src/robot.py` | DH transformations, robot configurations |
| `DataHandler` | `src/datahandler.py` | Data generation, normalization |
| `Trainer` | `src/trainer.py` | Training loop with validation |
| `IK_Solver` | Notebook 2 | Neural network model |
| `RobotAnimation` | `src/robot_animation.py` | 3D visualization |

### Data Flow

```
joints → Robot.buildDhTcpFrame() → TCP (forward kinematics)
TCP → IK_Solver (neural net) → predicted joints (inverse kinematics)
```

## Key Patterns

### Dual NumPy/PyTorch Support

The `use_torch` flag switches between NumPy and PyTorch implementations:

```python
robot.set_torch(True, device=device)  # Enable PyTorch mode
datahandler.set_torch(True)           # Moves tensors to GPU
```

### Normalization Modes

`normrange` controls output scaling using the `NormRange` enum from `src.normrange`:
- `NormRange.MINUS_ONE_TO_ONE` - Normalize to [-1, 1] range (matches Tanh activation)
- `NormRange.ZERO_TO_ONE` - Normalize to [0, 1] range

```python
from src.normrange import NormRange
datahandler.normrange = NormRange.MINUS_ONE_TO_ONE
```

### TCP Representation

`euler` flag controls orientation representation:
- `euler=True` - XYZ position + roll/pitch/yaw (6 values)
- `euler=False` - Full rotation matrix + XYZ (12 values)

### Joint Limits

Joint limits are stored in **degrees** but converted to **radians** internally:
```python
joint_limits = [[max_deg, min_deg], ...]  # Degrees in config
positions = np.radians(positions)          # Radians for computation
```

## Common Tasks

### Training a Model

```python
from src.datahandler import DataHandler
from src.robot import Robot
from src.trainer import Trainer
from src.normrange import NormRange

datahandler = DataHandler(robot=Robot(robot='youbot'))
datahandler.euler = False
datahandler.normrange = NormRange.MINUS_ONE_TO_ONE
tcp, gt_joints = datahandler(1000000)  # Generate training data

model = IK_Solver(in_shape, out_shape, neurons)
trainer = Trainer(model, optimizer, criterion, device)
trainer.tcp_loss = True  # Use TCP loss instead of joint loss
trainer(datahandler, samples=100000, epochs=20, batch_size=256)
```

### Adding a New Robot

Add configuration to `Robot.__init__()` in `src/robot.py`:

```python
elif robot == "myrobot":
    self.name = "MyRobot"
    mokuba = np.array([
        [0, 0, 0],      # Theta offsets
        [0.1, 0, 0.2],  # d values
        [0, 0.15, 0.1], # a values
        [np.pi/2, 0, 0] # alpha values
    ])
    joint_limits = [
        [180, -180],  # Joint 1 limits in degrees
        [90, -90],    # Joint 2 limits
        [120, -120]   # Joint 3 limits
    ]
```

### Hyperparameter Optimization

```python
import optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
```

See Notebook 2 for an example optimization setup.

## Important Notes

- **TCP loss mode**: When `trainer.tcp_loss = True`, training uses forward kinematics to compute predicted TCP and compares against input TCP. Only works with `relative=False` and `noised=False`.
- **Extreme positions**: `DataHandler` generates corner cases by default (`compute_extreme_positions=True`) to improve coverage of the joint space.
- **Batch processing**: `Robot.buildDhTcpFrame()` and `DataHandler` methods support batched operations for efficient GPU utilization.
- **Relative/noised modes**: For incremental IK or noisy training data, set `datahandler.relative=True` or `datahandler.noised=True`.

## File Dependencies

```
trainer.py
└── datahandler.py
    └── robot.py

robot_animation.py
└── robot.py
```
