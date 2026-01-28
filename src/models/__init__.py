"""Model architectures for inverse kinematics.

Public API:
    create_model(arch, **kwargs) - Create model by architecture name
    list_architectures() - List registered architectures
    save_checkpoint(model, path, ...) - Save model with full config
    load_checkpoint(path, device) - Load model from checkpoint
    load_checkpoint_full(path, device) - Load model with metadata
    IKModelBase - Base class for custom architectures
"""
from .registry import create_model, list_architectures, register
from .base import IKModelBase
from .checkpoint import save_checkpoint, load_checkpoint, load_checkpoint_full

# Import architectures to trigger registration
from . import mlp
from . import resmlp

__all__ = [
    "create_model",
    "list_architectures",
    "register",
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_full",
    "IKModelBase",
]
