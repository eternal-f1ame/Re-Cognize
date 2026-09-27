"""Backbone registry: the only place a backbone's facts are written down.

Consumed by the model builders, the training launcher, the evaluator and the weight
fetcher. `scripts/_config.py` re-exports it for the launchers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "reid_models" / "weights"


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    height: int
    width: int
    native_dim: int                          # BNNeck and memory block are sized to this; no adapters
    p: int = 8                               # PK identities per batch
    k: int = 4                               # PK crops per identity
    normalize: str = "imagenet"              # "imagenet" | "clip"
    patch_stride: Optional[int] = None       # ViT patch stride (TransReID released model: 12)
    hf_repo: Optional[str] = None
    hf_revision: Optional[str] = None
    weights: Optional[str] = None            # repo-relative path of the released checkpoint
    weights_sources: Tuple[str, ...] = ()    # "gdrive:<file id>" or https URL, tried in order
    weights_sha256: Optional[str] = None
    weights_note: str = ""
    reid5o_config: Optional[str] = None

    @property
    def feat_dim(self) -> int:               # alias of native_dim
        return self.native_dim

    @property
    def checkpoint(self) -> Optional[str]:   # alias of weights
        return self.weights

    def weights_path(self) -> Optional[Path]:
        return None if self.weights is None else REPO_ROOT / self.weights


BACKBONE_REGISTRY: Dict[str, BackboneConfig] = {
    "transreid": BackboneConfig(
        name="transreid", height=256, width=128, native_dim=768, p=8, k=4, normalize="imagenet", patch_stride=12,
        weights="reid_models/weights/transreid_vit_base_market1501.pth",
        weights_sources=("gdrive:11p4RjmpCGGAS-876VEt7OoFrUeHTUlyO",),
        weights_sha256="843055293e160da027d152e1b9cccb75e34c73e0a9714204012c0f2825f9c4e5",
        weights_note="damo-cv/TransReID model zoo, ViT-Base Market-1501 (stride 12, SIE, JPM); "
                     "second source: damo-cv/TransReID-SSL Market ViT-B; fallback: rename key to vit_b16_in21k",
    ),
    "magiv2": BackboneConfig(
        name="magiv2", height=224, width=224, native_dim=768, p=8, k=4, normalize="imagenet",
        hf_repo="ragavsachdeva/magiv2", hf_revision="fbc890fec52977142e8ee00bfe26e9458b65517c",
        weights_note="crop_embedding_model (ViT-MAE); mask_ratio forced to 0.0 at build",
    ),
    "magiv3": BackboneConfig(
        name="magiv3", height=384, width=384, native_dim=1024, p=4, k=4, normalize="imagenet",
        hf_repo="ragavsachdeva/magiv3", hf_revision="49c73a225122d53adbaa26d53868be81a57706e2",
        weights_note="Florence-2 image encoder; fp16 on CUDA, fp32 on CPU",
    ),
    "instructreid": BackboneConfig(
        name="instructreid", height=256, width=128, native_dim=768, p=8, k=4, normalize="imagenet", patch_stride=16,
        weights="reid_models/weights/instructreid_market.pth.tar",
        weights_sources=("gdrive:1w6nIPkbq-EPgTebAb9HViH-tmBKKpqiH",),
        weights_sha256="5309a1be56b7a85878db9e84e68f8866b9a75fe7fa3703b24a784ca58225fb8b",
        weights_note="hwz-zju/Instruct-ReID inference models, checkpoint_market.pth.tar; visual encoder only; "
                     "fallback: rename key to pass_vit_b with pass_vit_base_full.pth",
    ),
    "reid5o": BackboneConfig(
        name="reid5o", height=384, width=128, native_dim=512, p=8, k=4, normalize="clip",
        weights="reid_models/ReID5o/logs/reid5o_ckpt/best.pth",
        weights_sha256="49a7ee4c0b309af01f600b80ca456311badfac1ba8ea41bbb32a5d928c7b36df",
        weights_note="released ReID5o checkpoint (ReID5o project Google Drive); no download source listed, place it by hand",
        reid5o_config="reid_models/ReID5o/logs/reid5o_ckpt/configs.yaml",
    ),
}

ALL_BACKBONE_NAMES = list(BACKBONE_REGISTRY)
