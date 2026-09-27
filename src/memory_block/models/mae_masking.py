"""ViT-MAE masking control for the MagiV2 crop encoder.

The crop encoder is an HF ViTMAE model whose released config sets `mask_ratio=0.75`.
ViTMAE reads `config.mask_ratio` at forward time and masks that fraction of patches at
random on every forward pass, in eval mode too, so masking must be off for features to be
deterministic. Even at ratio 0 it gathers the tokens in argsort(rand) order, which changes
floating-point reduction order between forwards (measured 2e-7 on the CLS feature).
`disable_on_backbone` sets the ratio to 0 and pins the shuffle noise to the identity
permutation, so forwards are bit-identical.
"""
from __future__ import annotations

from typing import Optional

import torch


def pin_shuffle_noise(backbone) -> None:
    emb = getattr(backbone, "embeddings", None)
    if emb is None or not hasattr(emb, "random_masking") or getattr(emb, "_canonical_noise", False):
        return
    original = emb.random_masking

    def canonical_random_masking(sequence, noise=None, _orig=original):
        if noise is None:
            batch, length = sequence.shape[:2]
            noise = torch.arange(length, dtype=torch.float32, device=sequence.device).unsqueeze(0).expand(batch, length)
        return _orig(sequence, noise)

    emb.random_masking = canonical_random_masking
    emb._canonical_noise = True


def mask_ratio_of(backbone) -> Optional[float]:
    cfg = getattr(backbone, "config", None)
    return None if cfg is None or not hasattr(cfg, "mask_ratio") else float(cfg.mask_ratio)


def disable_on_backbone(backbone) -> Optional[float]:
    """Return the mask_ratio found (None if not a ViT-MAE), set it to 0.0, pin the shuffle, assert."""
    found = mask_ratio_of(backbone)
    if found is None:
        return None
    backbone.config.mask_ratio = 0.0
    if mask_ratio_of(backbone) != 0.0:
        raise RuntimeError("could not disable ViT-MAE masking")
    pin_shuffle_noise(backbone)
    return found


def _backbone_of(model):
    return getattr(getattr(model, "backbone_wrapper", None), "backbone", None)


def current_mask_ratio(model) -> Optional[float]:
    bb = _backbone_of(model)
    return None if bb is None else mask_ratio_of(bb)


def disable_mae_masking(model) -> Optional[float]:
    bb = _backbone_of(model)
    return None if bb is None else disable_on_backbone(bb)
