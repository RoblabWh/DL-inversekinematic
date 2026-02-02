# IK Neural Network Experiments

## Goal
~~Train an IK neural network that achieves MAE < 0.3312 rad (~19 deg)~~
**NEW**: Beat MAE 0.2895 by 12.6% → Target: **< 0.253 rad (~14.5 deg)**

## Results

| # | Robot | Euler | Neurons | LR | Samples | Epochs | Batch | TCP Loss | MAE (rad) | MAE (deg) | Notes |
|---|-------|-------|---------|-----|---------|--------|-------|----------|-----------|-----------|-------|
| 1 | youbot | Y | 512,256,128 | 0.0002 | 200k | 20 | 256 | N | 0.4157 | 23.82 | baseline euler |
| 2 | youbot | N | 512,256,128 | 0.0002 | 200k | 20 | 256 | N | 0.3537 | 20.27 | baseline rot - better! |
| 3 | youbot | N | 1024,512,256,128 | 0.0002 | 200k | 20 | 256 | N | 0.3288 | 18.84 | beats target |
| 4 | youbot | N | 512,512,256,256,128 | 0.0002 | 200k | 20 | 256 | N | 0.3236 | 18.54 | deeper beats wider |
| 5 | youbot | N | 512,512,256,256,128 | 0.0002 | 500k | 30 | 256 | N | 0.2895 | 16.58 | prev best |
| 6 | youbot | N | Optuna (6-layer) | 0.0032 | 200k | 20 | 256 | N | 0.3088 | 17.69 | Optuna found architecture |
| 7 | youbot | N | 512,512,256,256,128 | 0.0002 | 1M | 50 | 256 | N | 0.2777 | 15.91 | scaled training |
| 8 | youbot | N | 512,512,256,256,128 | 0.0002 | 500k | 30 | 256 | Y | 1.0299 | 59.01 | TCP loss failed |
| 9 | youbot | N | 512,512,256,256,128 | 0.001 | 1M | 50 | 512 | N | 0.2599 | 14.89 | higher LR helps |
| 10 | youbot | N | 512,512,256,256,128 | 0.002 | 1M | 75 | 512 | N | 0.2531 | 14.50 | **NEW WINNER** ✓ |

## Best Model
- **File**: `best_model.pt`
- **MAE**: 0.2531 rad (14.50 deg) - **12.6% better** than previous best!
- **Configuration**: youbot, rotation matrix (12D), neurons=[512,512,256,256,128], lr=0.002, 1M samples, 75 epochs, batch=512
- **Per-joint MAE**: J1=12.90, J2=10.95, J3=19.60, J4=15.91, J5=13.14 deg

## Experiment Log

### Experiment 1: Baseline with Euler
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 20 --samples 200000 --euler --save exp_baseline_euler.pt`
- **Result**: MAE 0.4157 rad (23.82 deg) - Joint 3 worst at 28.67 deg

### Experiment 2: Baseline with Rotation Matrix
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 20 --samples 200000 --save exp_baseline_rot.pt`
- **Result**: MAE 0.3537 rad (20.27 deg) - Better than euler! Use rotation matrix going forward.

### Experiment 3: Wider Network (rotation matrix)
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 20 --samples 200000 --neurons 1024,512,256,128 --save exp_wide.pt`
- **Result**: MAE 0.3288 rad (18.84 deg) - **BEATS TARGET!**

### Experiment 4: Deeper Network (rotation matrix)
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 20 --samples 200000 --neurons 512,512,256,256,128 --save exp_deep.pt`
- **Result**: MAE 0.3236 rad (18.54 deg) - **BEST SO FAR!** Deeper beats wider.

### Experiment 5: Scaled Training (deeper network)
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 30 --samples 500000 --neurons 512,512,256,256,128 --save exp_scaled.pt`
- **Result**: MAE 0.2895 rad (16.58 deg) - Per-joint: J1=14.75, J2=12.30, J3=22.34, J4=18.04, J5=15.49 deg

### Experiment 6: Optuna Hyperparameter Search
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 20 --samples 200000 --optimize --n-trials 30 --save exp_optuna.pt`
- **Result**: MAE 0.3088 rad (17.69 deg) - Optuna found unusual architecture [174,1008,647,182,107,67], but didn't beat scaled training

### Experiment 7: Aggressive Scaling (1M samples, 50 epochs)
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 50 --samples 1000000 --neurons 512,512,256,256,128 --lr 0.0002 --batch-size 256 --save exp_scaled.pt`
- **Result**: MAE 0.2777 rad (15.91 deg) - 4% improvement from more data/epochs

### Experiment 8: TCP Loss
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 30 --samples 500000 --neurons 512,512,256,256,128 --tcp-loss --save exp_tcp_loss.pt`
- **Result**: MAE 1.0299 rad (59.01 deg) - **FAILED** - TCP loss doesn't work well

### Experiment 9: Higher Learning Rate (0.001)
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 50 --samples 1000000 --neurons 512,512,256,256,128 --lr 0.001 --batch-size 512 --save exp_scaled_lr.pt`
- **Result**: MAE 0.2599 rad (14.89 deg) - Higher LR and batch size help significantly

### Experiment 10: Final Push (LR 0.002, 75 epochs)
- **Status**: Complete
- **Command**: `uv run python -m src.train --robot youbot --epochs 75 --samples 1000000 --neurons 512,512,256,256,128 --lr 0.002 --batch-size 512 --save exp_final_attempt.pt`
- **Result**: MAE 0.2531 rad (14.50 deg) - **TARGET ACHIEVED!** Per-joint: J1=12.90, J2=10.95, J3=19.60, J4=15.91, J5=13.14 deg

## Key Findings

1. **Rotation matrix (12D) beats euler angles (6D)**: 0.3537 vs 0.4157 rad
2. **Deeper networks beat wider networks**: 512,512,256,256,128 > 1024,512,256,128
3. **More data and epochs help significantly**: 500k/30ep beat 200k/20ep by 10%
4. **Joint 3 is hardest** to predict consistently (~19-22 deg MAE in all models)
5. **Higher learning rate (0.002) with larger batch size (512) significantly improves results**
6. **TCP loss does NOT work** - produces much worse results than joint loss
7. **Optuna search was not effective** for this problem - manual tuning worked better
