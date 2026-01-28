"""Base class for all IK models with normalization buffers."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn

from src.normrange import NormRange


class IKModelBase(nn.Module, ABC):
    """Abstract base class for IK solver models.

    Provides normalization bounds as model buffers that are automatically
    saved/loaded with state_dict and move with .to(device).
    """
    arch_name: str = "base"

    def __init__(self, in_features: int, out_features: int,
                 normalization_config: Dict[str, Any] | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._register_normalization_buffers(normalization_config)

    def _register_normalization_buffers(self, config: Dict[str, Any] | None):
        """Register normalization bounds as model buffers (saved with state_dict)."""
        if config is None:
            self.register_buffer('_norm_initialized', torch.tensor(False))
            self.register_buffer('x_min', torch.tensor(0.0))
            self.register_buffer('x_max', torch.tensor(1.0))
            self.register_buffer('y_min', torch.tensor(0.0))
            self.register_buffer('y_max', torch.tensor(1.0))
            self.register_buffer('z_min', torch.tensor(0.0))
            self.register_buffer('z_max', torch.tensor(1.0))
            self.register_buffer('_euler', torch.tensor(False))
            self.register_buffer('_normrange_minus1to1', torch.tensor(True))
        else:
            self.register_buffer('_norm_initialized', torch.tensor(True))
            self.register_buffer('x_min', torch.tensor(float(config['x_min'])))
            self.register_buffer('x_max', torch.tensor(float(config['x_max'])))
            self.register_buffer('y_min', torch.tensor(float(config['y_min'])))
            self.register_buffer('y_max', torch.tensor(float(config['y_max'])))
            self.register_buffer('z_min', torch.tensor(float(config['z_min'])))
            self.register_buffer('z_max', torch.tensor(float(config['z_max'])))
            self.register_buffer('_euler', torch.tensor(config.get('euler', False)))
            self.register_buffer('_normrange_minus1to1',
                               torch.tensor(config.get('normrange', NormRange.MINUS_ONE_TO_ONE.value) == NormRange.MINUS_ONE_TO_ONE.value))

    def set_normalization_bounds(self, config: Dict[str, Any]):
        """Update bounds (called by Trainer after first batch)."""
        self.x_min.fill_(float(config['x_min']))
        self.x_max.fill_(float(config['x_max']))
        self.y_min.fill_(float(config['y_min']))
        self.y_max.fill_(float(config['y_max']))
        self.z_min.fill_(float(config['z_min']))
        self.z_max.fill_(float(config['z_max']))
        self._euler.fill_(config.get('euler', False))
        self._normrange_minus1to1.fill_(config.get('normrange', NormRange.MINUS_ONE_TO_ONE.value) == NormRange.MINUS_ONE_TO_ONE.value)
        self._norm_initialized.fill_(True)

    def get_normalization_bounds(self) -> Dict[str, Any]:
        """Extract bounds for external use (e.g., DataHandler sync)."""
        return {
            'x_min': float(self.x_min.item()),
            'x_max': float(self.x_max.item()),
            'y_min': float(self.y_min.item()),
            'y_max': float(self.y_max.item()),
            'z_min': float(self.z_min.item()),
            'z_max': float(self.z_max.item()),
            'euler': bool(self._euler.item()),
            'normrange': NormRange.MINUS_ONE_TO_ONE.value if self._normrange_minus1to1.item() else NormRange.ZERO_TO_ONE.value,
        }

    @property
    def norm_initialized(self) -> bool:
        """Check if normalization bounds have been set."""
        return bool(self._norm_initialized.item())

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return config dict to reconstruct model (arch, in/out features, hyperparams)."""
        pass
