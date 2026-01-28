"""Checkpoint save/load with backward compatibility."""
from typing import Dict, Any
import torch
from .registry import create_model

CHECKPOINT_VERSION = 2


def save_checkpoint(
    model,
    path: str,
    robot_config: Dict[str, Any] | None = None,
    training_info: Dict[str, Any] | None = None,
):
    """Save model checkpoint with full configuration.

    The checkpoint includes:
    - Model config (architecture, hyperparameters)
    - State dict (weights + normalization buffers)
    - Robot config (optional)
    - Training info (optional)
    """
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),  # Includes normalization buffers!
        "robot_config": robot_config or {},
        "training_info": training_info or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device=None):
    """Load model from checkpoint with automatic architecture reconstruction.

    Supports both v2 (self-describing) and v1 (legacy) checkpoint formats.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_config" not in checkpoint:
        return _load_legacy_checkpoint(checkpoint, path, device)

    config = checkpoint["model_config"].copy()
    arch = config.pop("arch")
    model = create_model(arch, **config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def load_checkpoint_full(path: str, device=None) -> Dict[str, Any]:
    """Load checkpoint and return full metadata alongside the model.

    Returns a dict with keys: 'model', 'robot_config', 'training_info', 'version'
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_config" not in checkpoint:
        model = _load_legacy_checkpoint(checkpoint, path, device)
        return {
            "model": model,
            "robot_config": {},
            "training_info": {},
            "version": 1,
        }

    config = checkpoint["model_config"].copy()
    arch = config.pop("arch")
    model = create_model(arch, **config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    return {
        "model": model,
        "robot_config": checkpoint.get("robot_config", {}),
        "training_info": checkpoint.get("training_info", {}),
        "version": checkpoint.get("checkpoint_version", 2),
    }


def _load_legacy_checkpoint(checkpoint: Dict[str, Any], path: str, device) -> torch.nn.Module:
    """Handle v1 format: {'model_state_dict', 'normalization_bounds', 'neurons'} or plain state_dict."""
    # Determine if it's a dict-wrapped checkpoint or plain state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        neurons = checkpoint.get("neurons", [512, 256, 128])
        norm_bounds = checkpoint.get("normalization_bounds")
    else:
        # Plain state_dict
        state_dict = checkpoint
        neurons = [512, 256, 128]  # Default fallback
        norm_bounds = None

    # Infer dimensions from state_dict
    # Find first linear layer weight to get in_features
    first_weight_key = None
    for key in state_dict.keys():
        if "weight" in key and len(state_dict[key].shape) == 2:
            first_weight_key = key
            break

    if first_weight_key is None:
        raise ValueError(f"Cannot infer model dimensions from legacy checkpoint: {path}")

    first_weight = state_dict[first_weight_key]
    in_features = first_weight.shape[1]

    # Find last linear layer to get out_features
    last_weight_key = None
    for key in reversed(list(state_dict.keys())):
        if "weight" in key and len(state_dict[key].shape) == 2:
            last_weight_key = key
            break

    last_weight = state_dict[last_weight_key]
    out_features = last_weight.shape[0]

    # Infer neurons from intermediate layer sizes
    inferred_neurons = []
    for key in state_dict.keys():
        if "weight" in key and len(state_dict[key].shape) == 2:
            out_dim = state_dict[key].shape[0]
            if out_dim != out_features:  # Skip output layer
                inferred_neurons.append(out_dim)

    # Use inferred neurons if we found them, otherwise use provided/default
    if inferred_neurons:
        neurons = inferred_neurons

    # Detect if batchnorm was used
    use_batchnorm = any("BatchNorm" in key or "running_mean" in key for key in state_dict.keys())

    model = create_model(
        "mlp",
        in_features=in_features,
        out_features=out_features,
        neurons=neurons,
        use_batchnorm=use_batchnorm,
        normalization_config=norm_bounds,
    )

    # Load state dict with strict=False to handle potential mismatches
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model
