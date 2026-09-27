"""
Training Module for Memory Block

Contains training-specific components:
- Memory-specific loss functions
- Training utilities
"""

from .losses import (
    MemoryConsistencyLoss,
    CombinedMemoryLoss,
)
from .train import train_memory_model

__all__ = [
    "MemoryConsistencyLoss",
    "CombinedMemoryLoss",
    "train_memory_model",
]
