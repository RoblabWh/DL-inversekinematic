#!/usr/bin/env python3
"""
Training script for IK neural network.

Usage:
    # Train a new model
    python -m src.train --robot youbot --epochs 5 --samples 50000 --euler

    # Test an existing model
    python -m src.train --load model.pt --test-only --euler

    # Custom architecture
    python -m src.train --neurons 1024,512,256,128 --epochs 10 --euler

    # Choose architecture and activations
    python -m src.train --arch mlp --activation relu --output-activation sigmoid --epochs 5

    # Configure data handling
    python -m src.train --normrange zero_to_one --no-extreme-positions --validation-split 0.1

    # Train with noised data
    python -m src.train --noised --sigma 3.0 --epochs 5

    # Hyperparameter optimization with Optuna
    python -m src.train --optimize --n-trials 10 --epochs 5 --euler
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.datahandler import DataHandler
from src.robot import Robot
from src.trainer import Trainer
from src.models import create_model, list_architectures, save_checkpoint, load_checkpoint
from src.normrange import NormRange
from src.metrics import compute_all_metrics, format_metrics

ACTIVATION_CHOICES = ["relu", "gelu", "tanh", "sigmoid", "leaky_relu", "elu", "selu", "none"]


def _create_optimizer(model, args):
    """Create optimizer based on CLI args."""
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def _create_criterion(args):
    """Create loss function based on CLI args."""
    if args.loss == "huber":
        return nn.HuberLoss()
    elif args.loss == "l1":
        return nn.L1Loss()
    return nn.MSELoss()


def _scheduler_kwargs(args):
    """Build scheduler kwargs from CLI args."""
    kwargs = {}
    if args.scheduler == "cosine":
        kwargs["T_0"] = args.cosine_t0
        kwargs["T_mult"] = args.cosine_tmult
    elif args.scheduler == "onecycle":
        kwargs["max_lr"] = args.lr
    return kwargs


def train(model: nn.Module, datahandler: DataHandler, args, robot) -> float:
    """Train the model and return the final validation loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = _create_optimizer(model, args)
    criterion = _create_criterion(args)

    datahandler.set_torch(True)
    trainer = Trainer(
        model, optimizer, criterion, device,
        scheduler_type=args.scheduler,
        scheduler_kwargs=_scheduler_kwargs(args),
        grad_clip=args.grad_clip,
        trackio_project=args.trackio_project,
        trackio_space_id=args.trackio_space_id,
        compute_runtime_metrics=args.compute_runtime_metrics,
        eval_interval=args.eval_interval,
        log_batch_loss=args.log_batch_loss,
        run_name=args.run_name,
        compute_llc=args.compute_llc,
        llc_interval=args.llc_interval,
        llc_num_chains=args.llc_num_chains,
        llc_num_draws=args.llc_num_draws,
        llc_num_burnin_steps=args.llc_num_burnin_steps,
        llc_num_steps_bw_draws=args.llc_num_steps_bw_draws,
    )
    trainer.tcp_loss = args.tcp_loss

    trainer(datahandler, samples=args.samples, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split)

    # Save the model with new checkpoint format
    save_checkpoint(
        model,
        args.save,
        robot_config={"name": robot.name, "joint_limits": robot.joint_limits},
        training_info={"epochs": args.epochs, "samples": args.samples, "batch_size": args.batch_size},
    )
    print(f"Model saved to {args.save}")

    # Return final validation loss by running one more pass
    _, val_loader = datahandler.get_data_loaders(args.samples // 10, args.batch_size, validation_split=args.validation_split)
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_loader)
    return val_loss


def test(model: nn.Module, datahandler: DataHandler, args) -> dict:
    """Run model testing and return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    datahandler.compute_extreme_positions = False
    datahandler.set_torch(True)

    tcp, gt_joints = datahandler(args.test_samples)
    with torch.no_grad():
        pred = model(tcp)

    # Denormalize and convert to numpy
    gt_pos = datahandler.denorm_joint(gt_joints).cpu().numpy()
    pred_pos = datahandler.denorm_joint(pred).cpu().numpy()

    metrics = compute_all_metrics(gt_pos, pred_pos, robot=datahandler.robot)
    print("\n" + format_metrics(metrics))
    return metrics


def objective(trial, datahandler: DataHandler, args) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 2, 6)
    neurons = []
    for i in range(n_layers):
        neurons.append(trial.suggest_int(f"n_units_l{i}", 64, 2048, log=True))

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    activation = trial.suggest_categorical("activation", ACTIVATION_CHOICES[:6])  # exclude selu/none
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Create model
    in_features = datahandler.get_input_shape()
    out_features = datahandler.get_output_shape()
    model = create_model(
        args.arch,
        in_features=in_features,
        out_features=out_features,
        neurons=neurons,
        activation=activation,
        output_activation=args.output_activation,
        use_batchnorm=not args.no_batchnorm,
        dropout=dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    datahandler.set_torch(True)
    trainer = Trainer(model, optimizer, criterion, device)
    trainer.tcp_loss = args.tcp_loss

    # Train for reduced epochs
    reduced_epochs = max(1, args.epochs // 4)
    reduced_samples = max(50000, args.samples // 4)

    trainer(datahandler, samples=reduced_samples, epochs=reduced_epochs, batch_size=batch_size)
    model.eval()

    # Calculate validation loss
    _, val_loader = datahandler.get_data_loaders(20000, batch_size)
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_loader)

    return val_loss


def optimize(datahandler: DataHandler, args):
    """Run Optuna hyperparameter optimization."""
    try:
        import optuna
    except ImportError:
        print("Optuna is not installed. Install it with: pip install optuna")
        return

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, datahandler, args), n_trials=args.n_trials)

    print("\n--- Optimization Results ---")
    print(f"Best trial value: {study.best_trial.value:.6f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Retrain with best parameters
    print("\n--- Retraining with best parameters ---")
    best_params = study.best_trial.params
    neurons = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]

    in_features = datahandler.get_input_shape()
    out_features = datahandler.get_output_shape()
    model = create_model(
        args.arch,
        in_features=in_features,
        out_features=out_features,
        neurons=neurons,
        activation=best_params.get("activation", args.activation),
        output_activation=args.output_activation,
        use_batchnorm=not args.no_batchnorm,
        dropout=best_params.get("dropout", args.dropout),
    )

    # Override args with best params
    args.lr = best_params["lr"]
    args.batch_size = best_params["batch_size"]

    train(model, datahandler, args, datahandler.robot)
    test(model, datahandler, args)


def verify_gradients(model: nn.Module, datahandler: DataHandler) -> bool:
    """Verify gradients flow through TCP loss computation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    datahandler.set_torch(True)
    datahandler.compute_extreme_positions = False

    # Generate enough data to initialize normalization bounds
    tcp, joints = datahandler(10000)

    # Forward pass through model
    pred = model(tcp)

    # TCP loss computation chain
    denorm = datahandler.denorm_joint(pred)
    pred_tcp = datahandler.get_tcp(denorm)
    norm_tcp = datahandler.norm_tcp(pred_tcp)

    # Compute loss and backward
    loss = ((norm_tcp - tcp) ** 2).mean()
    loss.backward()

    # Check if gradients flowed to model parameters
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    print(f"Gradients flowing: {has_grad}")
    if has_grad:
        # Show gradient stats for first layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad norm = {grad_norm:.6f}")
                break  # Just show the first one as a sample

    return has_grad


def parse_neurons(s: str) -> list[int]:
    """Parse comma-separated neuron counts."""
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Train and test IK neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="youbot",
        choices=["youbot", "twoaxis", "threeaxis", "fouraxis", "fiveaxis", "baxter"],
        help="Robot type",
    )
    parser.add_argument("--samples", type=int, default=200000, help="Training samples per epoch")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=250, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument(
        "--neurons",
        type=str,
        default="512,256,128",
        help="Hidden layer sizes (comma-separated)",
    )
    parser.add_argument(
        "--euler",
        action="store_true",
        help="Use euler angles (6D) instead of rotation matrix (12D)",
    )
    parser.add_argument(
        "--tcp-loss",
        action="store_true",
        help="Use TCP loss instead of joint loss",
    )
    parser.add_argument("--save", type=str, default="model.pt", help="Path to save trained model")
    parser.add_argument("--load", type=str, default=None, help="Path to load existing model")
    parser.add_argument("--test-only", action="store_true", help="Skip training, only run testing")
    parser.add_argument("--test-samples", type=int, default=20000, help="Number of samples for testing")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument(
        "--verify-gradients",
        action="store_true",
        help="Verify gradients flow through TCP loss computation and exit",
    )

    # DataHandler arguments
    parser.add_argument(
        "--normrange",
        type=str,
        default="minus1to1",
        choices=["minus1to1", "zero_to_one"],
        help="Normalization range (minus1to1 → [-1,1], zero_to_one → [0,1])",
    )
    parser.add_argument("--relative", action="store_true", help="Use relative TCP positions")
    parser.add_argument("--noised", action="store_true", help="Use noised target positions")
    parser.add_argument("--sigma", type=float, default=2.0, help="Noise std dev in degrees (only with --noised)")
    parser.add_argument(
        "--no-extreme-positions",
        action="store_true",
        help="Disable generation of extreme (corner-case) positions",
    )
    parser.add_argument("--split", type=int, default=4, help="Grid resolution for extreme positions")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Fraction of data used for validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose DataHandler logging")

    # Model architecture arguments
    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        help="Model architecture (available: %(default)s)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=ACTIVATION_CHOICES,
        help="Hidden layer activation function",
    )
    parser.add_argument(
        "--output-activation",
        type=str,
        default="tanh",
        choices=ACTIVATION_CHOICES,
        help="Output layer activation function",
    )
    parser.add_argument("--no-batchnorm", action="store_true", help="Disable BatchNorm in hidden layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0.0 = no dropout)")

    # ResidualMLP-specific arguments
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for resmlp architecture")
    parser.add_argument("--num-blocks", type=int, default=6, help="Number of residual blocks for resmlp architecture")

    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Optimizer type",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 regularization)")

    # Scheduler arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "onecycle"],
        help="Learning rate scheduler type",
    )
    parser.add_argument("--cosine-t0", type=int, default=10, help="CosineAnnealingWarmRestarts T_0")
    parser.add_argument("--cosine-tmult", type=int, default=2, help="CosineAnnealingWarmRestarts T_mult")

    # Gradient clipping
    parser.add_argument("--grad-clip", type=float, default=None, help="Max gradient norm for clipping (None = no clipping)")

    # Trackio integration
    parser.add_argument("--trackio-project", type=str, default=None, help="Trackio project name (enables experiment tracking)")
    parser.add_argument("--trackio-space-id", type=str, default=None, help="Trackio Space ID for syncing")
    parser.add_argument("--compute-runtime-metrics", action="store_true", help="Compute detailed metrics during training")
    parser.add_argument("--eval-interval", type=int, default=5, help="Epochs between runtime metric evaluations")
    parser.add_argument("--log-batch-loss", action="store_true", help="Log per-batch loss to Trackio")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name for checkpoints")

    # LLC/Grokking detection
    parser.add_argument("--compute-llc", action="store_true", help="Enable LLC grokking detection")
    parser.add_argument("--llc-interval", type=int, default=10, help="Epochs between LLC computations")
    parser.add_argument("--llc-num-chains", type=int, default=3, help="Number of MCMC chains for LLC")
    parser.add_argument("--llc-num-draws", type=int, default=100, help="Samples per chain for LLC")
    parser.add_argument("--llc-num-burnin-steps", type=int, default=200, help="Burn-in steps before sampling (should be >= num-draws)")
    parser.add_argument("--llc-num-steps-bw-draws", type=int, default=5, help="Steps between draws")

    # Loss function
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber", "l1"],
        help="Loss function",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.tcp_loss and (args.relative or args.noised):
        parser.error("--tcp-loss is incompatible with --relative and --noised")

    # Validate architecture name
    available_archs = list_architectures()
    if args.arch not in available_archs:
        parser.error(f"Unknown architecture: {args.arch}. Available: {available_archs}")

    # Map normrange string to enum
    normrange_map = {"minus1to1": NormRange.MINUS_ONE_TO_ONE, "zero_to_one": NormRange.ZERO_TO_ONE}
    normrange = normrange_map[args.normrange]

    # Initialize robot and datahandler
    robot = Robot(robot=args.robot)
    datahandler = DataHandler(robot=robot)
    datahandler.euler = args.euler
    datahandler.relative = args.relative
    datahandler.noised = args.noised
    datahandler.normrange = normrange
    datahandler.compute_extreme_positions = not args.no_extreme_positions
    datahandler.split = args.split
    datahandler.verbose = args.verbose

    print(f"Robot: {robot.name}")
    print(f"Using euler angles: {args.euler}")
    print(f"Normalization range: {normrange.name}")
    print(f"Relative: {args.relative}, Noised: {args.noised}" + (f" (sigma={args.sigma})" if args.noised else ""))
    print(f"Extreme positions: {not args.no_extreme_positions} (split={args.split})")
    print(f"Architecture: {args.arch}, Activation: {args.activation}, Output: {args.output_activation}")
    print(f"BatchNorm: {not args.no_batchnorm}, Dropout: {args.dropout}")
    print(f"Optimizer: {args.optimizer}, Scheduler: {args.scheduler}, Loss: {args.loss}")
    print(f"Weight decay: {args.weight_decay}, Grad clip: {args.grad_clip}")
    if args.trackio_project:
        print(f"Trackio: {args.trackio_project}" + (f" (space: {args.trackio_space_id})" if args.trackio_space_id else ""))
    if args.compute_llc:
        print(f"LLC detection: enabled (interval={args.llc_interval}, chains={args.llc_num_chains}, draws={args.llc_num_draws}, burnin={args.llc_num_burnin_steps})")
    print(f"Input shape: {datahandler.get_input_shape()}")
    print(f"Output shape: {datahandler.get_output_shape()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.optimize:
        optimize(datahandler, args)
        return

    # Create or load model
    neurons = parse_neurons(args.neurons)
    in_features = datahandler.get_input_shape()
    out_features = datahandler.get_output_shape()

    if args.load:
        model = load_checkpoint(args.load, device=device)
        # Sync normalization bounds from model to datahandler for inference
        if model.norm_initialized:
            datahandler.set_normalization_bounds(model.get_normalization_bounds())
        print(f"Loaded model from {args.load}")
        print(f"Loaded architecture: {model.get_config()}")
    elif args.arch == "resmlp":
        model = create_model(
            args.arch,
            in_features=in_features,
            out_features=out_features,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            activation=args.activation,
            output_activation=args.output_activation,
            dropout=args.dropout,
        )
        print(f"Model architecture: resmlp hidden_dim={args.hidden_dim} blocks={args.num_blocks}")
    else:
        model = create_model(
            args.arch,
            in_features=in_features,
            out_features=out_features,
            neurons=neurons,
            activation=args.activation,
            output_activation=args.output_activation,
            use_batchnorm=not args.no_batchnorm,
            dropout=args.dropout,
        )
        print(f"Model architecture: {neurons}")

    print(model)

    if args.verify_gradients:
        success = verify_gradients(model, datahandler)
        return 0 if success else 1

    if not args.test_only:
        train(model, datahandler, args, robot)

    test(model, datahandler, args)


if __name__ == "__main__":
    main()
