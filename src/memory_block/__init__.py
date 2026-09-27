"""
Memory Block Module for Manga Character Re-Identification

This module implements a memory-enhanced architecture for manga character Re-ID,
combining working and episodic memory mechanisms on top of a frozen backbone.

Architecture:
- Working Memory: Short-term context from recent panels (sliding window)
- Episodic Memory: Long-term character identity bank (non-parametric prototypes)
- Gated Fusion: Adaptive combination of memory outputs

Usage:
    from memory_block.models import MemoryEnhancedReID, MemoryConfig

    config = MemoryConfig(num_classes=100)
    model = MemoryEnhancedReID(config)
"""

from .models import (
    MemoryEnhancedReID,
    MemoryConfig,
    BackboneWrapper,
    WorkingMemory,
    EpisodicMemory,
    GatedFusion,
)
from .models.memory_modules import MemoryBlock

__all__ = [
    # Main classes
    "MemoryEnhancedReID",
    "MemoryConfig",
    "BackboneWrapper",
    # Memory modules
    "WorkingMemory",
    "EpisodicMemory",
    "GatedFusion",
    "MemoryBlock",
]
