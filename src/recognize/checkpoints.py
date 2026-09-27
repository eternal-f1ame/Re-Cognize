"""Loading trained checkpoints for evaluation.

A checkpoint written by `memory_block.training.train` carries its own config. It is built from
that config with the current builders, so it has the weights, patch stride and normalisation it was
trained with, and its state dict is loaded strictly.

ViT-MAE patch masking is switched off and asserted, since masking draws new random patches at every
forward pass; the ratio found in the checkpoint is recorded as `mask_ratio_at_training`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from memory_block.models.mae_masking import current_mask_ratio, disable_mae_masking  # re-exported

from .loading import load_state_dict_checked


@dataclass
class LoadedModel:
    model: nn.Module
    backbone: str
    checkpoint: Optional[Path]
    normalize: str                      # "imagenet" | "clip"
    flags: Dict[str, Any] = field(default_factory=dict)


def _default_construct(config):
    from memory_block.models.model import MemoryEnhancedReID
    return MemoryEnhancedReID(config)


def load_checkpoint(path, device: str = "cpu", construct: Optional[Callable] = None) -> LoadedModel:
    """Build a trained checkpoint from its own config with the current builders and load it strictly."""
    from memory_block.models.model import MemoryConfig
    from .backbones import BACKBONE_REGISTRY
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "model_state_dict" not in ckpt:
        raise RuntimeError(f"{path} is not a training checkpoint (it has no config or model_state_dict)")
    config = ckpt["config"]
    if not isinstance(config, MemoryConfig):
        config = MemoryConfig(**{k: v for k, v in dict(config).items() if k in MemoryConfig.__dataclass_fields__})
    model = (construct or _default_construct)(config)
    load_state_dict_checked(model, ckpt["model_state_dict"])
    mask_found = disable_mae_masking(model)
    model = model.to(device).eval()
    spec = BACKBONE_REGISTRY.get(config.backbone_type)
    args = ckpt.get("args", {})
    flags = {
        "training_mode": ckpt.get("training_mode"),
        "lora": bool(getattr(config, "use_lora", False)),
        "memory": getattr(model, "memory_block", None) is not None,
        "mask_ratio_at_training": mask_found,
        "mask_ratio": current_mask_ratio(model),
        "config_label": args.get("name"),
        "tag": args.get("name") or path.parent.name,
        "run_epoch": ckpt.get("epoch"),
    }
    normalize = args.get("normalize") or (spec.normalize if spec else "imagenet")
    print(f"[checkpoints] {path}: {config.backbone_type} epoch {ckpt.get('epoch')} "
          f"memory={flags['memory']} lora={flags['lora']} normalize={normalize}")
    return LoadedModel(model=model, backbone=config.backbone_type, checkpoint=path, normalize=normalize, flags=flags)
