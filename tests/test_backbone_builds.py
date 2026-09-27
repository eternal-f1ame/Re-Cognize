"""Every backbone builds from the registry with released weights, strictly and deterministically (slow)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from recognize.backbones import BACKBONE_REGISTRY

pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("ignore::FutureWarning")]


def _build(name):
    import evaluate
    cfg = BACKBONE_REGISTRY[name]
    if cfg.weights and not cfg.weights_path().exists():
        pytest.skip(f"{name}: released weights not fetched")
    if name == "magiv3" and not os.environ.get("RUN_SLOW_MAGIV3"):
        pytest.skip("set RUN_SLOW_MAGIV3=1 to build Florence-2 on CPU")
    torch.set_num_threads(8)
    return evaluate.build_model(SimpleNamespace(checkpoint=None, pretrained=name, memory=None), "cpu"), cfg


@pytest.mark.parametrize("name", ["transreid", "instructreid", "reid5o", "magiv2", "magiv3"])
def test_registry_build_is_strict_native_and_deterministic(name):
    lm, cfg = _build(name)
    w = lm.model.backbone_wrapper
    assert w.feat_dim == cfg.native_dim and lm.normalize == cfg.normalize
    x = torch.rand(2, 3, cfg.height, cfg.width)
    outs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            outs.append(lm.model(x, use_memory=False)["bn_feat"])
    assert torch.equal(outs[0], outs[1]) and outs[0].shape == (2, cfg.native_dim)
    if name == "transreid":
        assert w.pos_grid == (21, 10) and w.backbone.patch_embed.proj.stride == (12, 12)
        with torch.no_grad():
            patches, cls = w(x)
        assert patches.shape[1] == 210
    if name == "magiv2":
        assert w.backbone.config.mask_ratio == 0.0


@pytest.mark.parametrize("name", ["transreid", "instructreid", "reid5o", "magiv2", "magiv3"])
def test_lora_adapters_all_sit_on_the_image_path(name):
    """No injected adapter may be dead: it would make the +LoRA row differ from its base row by noise."""
    from memory_block.models.lora import LoRALinear
    from memory_block.models.model import MemoryConfig, MemoryEnhancedReID
    cfg = BACKBONE_REGISTRY[name]
    if cfg.weights and not cfg.weights_path().exists():
        pytest.skip(f"{name}: released weights not fetched")
    if name == "magiv3" and not os.environ.get("RUN_SLOW_MAGIV3"):
        pytest.skip("set RUN_SLOW_MAGIV3=1 to build Florence-2 on CPU")
    torch.set_num_threads(8)
    model = MemoryEnhancedReID(MemoryConfig(
        backbone_type=name, num_classes=3, feat_dim=cfg.native_dim, freeze_backbone=True, use_lora=True,
        lora_rank=8, lora_alpha=16.0, lora_layers=4, image_height=cfg.height, image_width=cfg.width)).eval()
    adapters = {n for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    assert adapters, f"{name}: LoRA requested but nothing was injected"
    fired = set()
    hooks = [m.register_forward_hook(lambda mo, i, o, n=n: fired.add(n))
             for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    with torch.no_grad():
        model(torch.rand(1, 3, cfg.height, cfg.width), use_memory=False)
    for h in hooks:
        h.remove()
    assert fired == adapters, f"{name}: dead adapters {sorted(adapters - fired)[:3]}"
