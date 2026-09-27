"""Key maps from released Re-ID checkpoints to the backbone modules used here.

Every load is strict after the map; anything dropped is listed explicitly.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

# damo-cv/TransReID checkpoint layout (ViT-Base, Market-1501): backbone under "base.", plus
# "base.sie_embed" (camera/view embeddings, no cameras in manga), "base.fc.*" (ImageNet head),
# "b1.*"/"b2.*" (JPM local branches), "bottleneck*"/"classifier*" (Re-ID heads).
TRANSREID_PREFIX = "base."
TRANSREID_DROP = ("sie_embed", "fc.", "head.")

# hwz-zju/Instruct-ReID checkpoint: the visual encoder (PASS ViT-B) under "module.visual_encoder.";
# "module.visual_encoder_m." is the momentum copy; text encoder, fusion, heads and queues are dropped.
INSTRUCTREID_PREFIX = "module.visual_encoder."


def map_transreid_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if not k.startswith(TRANSREID_PREFIX):
            continue
        k2 = k[len(TRANSREID_PREFIX):]
        if k2.startswith(TRANSREID_DROP):
            continue
        out[k2] = v
    return out


def map_instructreid_keys(sd: Dict[str, torch.Tensor], prefix: str = INSTRUCTREID_PREFIX) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def infer_pos_grid(n_tokens: int, img: Tuple[int, int], patch: int, stride: int) -> Tuple[int, int]:
    """Patch grid (gh, gw) of a position embedding with `n_tokens` rows (cls included)."""
    gh = (img[0] - patch) // stride + 1
    gw = (img[1] - patch) // stride + 1
    if gh * gw != n_tokens - 1:
        raise ValueError(f"pos_embed has {n_tokens - 1} patch tokens but img {img}, patch {patch}, stride {stride} "
                         f"gives {gh}x{gw}={gh * gw}")
    return gh, gw
