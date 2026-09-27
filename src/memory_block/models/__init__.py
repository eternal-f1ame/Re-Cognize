"""
Memory Block Models

Contains:
- MemoryEnhancedReID: Universal memory-enhanced Re-ID model (works with any backbone)
- WorkingMemory: Short-term panel context memory
- EpisodicMemory: Long-term character identity bank
- GatedFusion: Adaptive memory fusion
- MemoryConfig: Configuration dataclass
- BackboneWrapper: Universal backbone interface

Supported Backbones:
- TransReID (ViT-B/16, 768-dim)
- MagiV2 (ViT-based, 768-dim)
- MagiV3 (Florence2, 1024-dim)
- InstructReID (ViT-B/16, 768-dim)
- ReID5o (CLIP ViT-B/16, 512-dim)
"""

from .memory_modules import WorkingMemory, EpisodicMemory, GatedFusion
from .model import (
    MemoryEnhancedReID,
    MemoryConfig,
    BackboneWrapper,
)

__all__ = [
    "MemoryEnhancedReID",
    "MemoryConfig",
    "BackboneWrapper",
    "WorkingMemory",
    "EpisodicMemory",
    "GatedFusion",
]
