"""Residual MLP architecture for inverse kinematics."""
from typing import Dict, Any
import torch
import torch.nn as nn
from .base import IKModelBase
from .registry import register


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN -> Act -> Linear -> BN -> Act -> Linear + skip."""

    def __init__(self, dim: int, activation: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = activation
        self.linear1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = type(activation)()  # fresh instance for second activation
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn1(x)
        out = self.act1(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + residual


@register("resmlp")
class ResidualMLPModel(IKModelBase):
    """Residual MLP with skip connections for inverse kinematics.

    Architecture:
        Input projection: Linear(in, hidden) -> BN -> Act
        N x ResidualBlock(hidden)
        Output projection: BN -> Act -> Linear(hidden, out) -> OutputAct
    """
    arch_name = "resmlp"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        activation: str = "gelu",
        dropout: float = 0.0,
        output_activation: str = "tanh",
        normalization_config: Dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(in_features, out_features, normalization_config)
        self._hidden_dim = hidden_dim
        self._num_blocks = num_blocks
        self._activation = activation
        self._dropout = dropout
        self._output_activation = output_activation

        act_fn = self._get_activation(activation)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            type(act_fn)(),
        )

        # Residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, self._get_activation(activation), dropout)
              for _ in range(num_blocks)]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            self._get_activation(activation),
            nn.Linear(hidden_dim, out_features),
            self._get_activation(output_activation),
        )

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "none": nn.Identity,
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        return activations[name.lower()]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "arch": self.arch_name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_dim": self._hidden_dim,
            "num_blocks": self._num_blocks,
            "activation": self._activation,
            "dropout": self._dropout,
            "output_activation": self._output_activation,
        }
