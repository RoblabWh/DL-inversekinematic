"""MLP architecture - migration of current IK_Solver."""
from typing import Dict, Any, List
import torch.nn as nn
from .base import IKModelBase
from .registry import register


@register("mlp")
class MLPModel(IKModelBase):
    """Configurable MLP for inverse kinematics.

    This is a migration of the original IK_Solver class with additional
    configurability for activation functions, dropout, and output activation.
    """
    arch_name = "mlp"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        neurons: List[int],
        activation: str = "gelu",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        output_activation: str = "tanh",
        normalization_config: Dict[str, Any] | None = None,
    ):
        super().__init__(in_features, out_features, normalization_config)
        self._neurons = neurons
        self._activation = activation
        self._use_batchnorm = use_batchnorm
        self._dropout = dropout
        self._output_activation = output_activation

        layers = []
        input_dim = in_features
        for out_dim in neurons:
            layers.append(nn.Linear(input_dim, out_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = out_dim
        layers.append(nn.Linear(input_dim, out_features))
        layers.append(self._get_activation(output_activation))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation module by name."""
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

    def forward(self, x):
        return self.layers(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            "arch": self.arch_name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "neurons": self._neurons,
            "activation": self._activation,
            "use_batchnorm": self._use_batchnorm,
            "dropout": self._dropout,
            "output_activation": self._output_activation,
        }
