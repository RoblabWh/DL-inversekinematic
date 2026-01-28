"""Normalization range enum for data scaling."""
from enum import Enum


class NormRange(Enum):
    """Normalization range for data scaling.

    MINUS_ONE_TO_ONE: Maps to [-1, 1], matches Tanh activation
    ZERO_TO_ONE: Maps to [0, 1]
    """

    MINUS_ONE_TO_ONE = "minus1to1"
    ZERO_TO_ONE = "zeroToOne"

    @classmethod
    def from_string(cls, s: str) -> "NormRange":
        """Parse from legacy string format (for checkpoint compatibility)."""
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown normrange: {s}. Valid: {[m.value for m in cls]}")
