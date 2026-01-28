"""Model registry with @register decorator."""
from typing import Type, Dict, Callable
import torch.nn as nn

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register(name: str) -> Callable:
    """Decorator to register an architecture class."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def create_model(arch: str, **kwargs) -> nn.Module:
    """Factory function to create model by architecture name."""
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[arch](**kwargs)


def list_architectures() -> list[str]:
    """Return list of registered architecture names."""
    return list(_REGISTRY.keys())
