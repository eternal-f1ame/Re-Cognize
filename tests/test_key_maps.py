"""Key maps for the released TransReID and Instruct-ReID checkpoints."""
from __future__ import annotations

import pytest
import torch


def _fake_transreid_state():
    from timm.models.vision_transformer import vit_base_patch16_224
    ref = vit_base_patch16_224(pretrained=False); ref.reset_classifier(0)
    sd = {"base." + k: v for k, v in ref.state_dict().items()}
    sd["base.pos_embed"] = torch.zeros(1, 211, 768)               # 21x10 grid at 256x128, stride 12
    sd["base.sie_embed"] = torch.zeros(6, 1, 768)
    sd["base.fc.weight"] = torch.zeros(1000, 768); sd["base.fc.bias"] = torch.zeros(1000)
    for br in ("b1", "b2"):
        sd[f"{br}.0.norm1.weight"] = torch.zeros(768)
    for h in ("bottleneck", "bottleneck_1", "classifier", "classifier_1"):
        sd[f"{h}.weight"] = torch.zeros(768)
    return sd, set(ref.state_dict())


class TestTransReID:
    def test_map_yields_exactly_the_timm_keys(self):
        from memory_block.models.key_maps import map_transreid_keys
        sd, expected = _fake_transreid_state()
        out = map_transreid_keys(sd)
        assert set(out) == expected and out["pos_embed"].shape == (1, 211, 768)
        assert not any(k.startswith(("sie", "fc.", "b1", "bottleneck", "classifier")) for k in out)

    def test_pos_grid(self):
        from memory_block.models.key_maps import infer_pos_grid
        assert infer_pos_grid(211, (256, 128), 16, 12) == (21, 10)
        assert infer_pos_grid(129, (256, 128), 16, 16) == (16, 8)
        with pytest.raises(ValueError):
            infer_pos_grid(197, (256, 128), 16, 12)


class TestInstructReID:
    def test_keeps_visual_encoder_only(self):
        from memory_block.models.key_maps import map_instructreid_keys
        sd = {"module.visual_encoder.cls_token": 1, "module.visual_encoder.blocks.0.attn.qkv.weight": 2,
              "module.visual_encoder_m.cls_token": 3, "module.text_encoder.x": 4, "module.classifier.weight": 5,
              "module.visual_encoder.fc.weight": 6}
        assert map_instructreid_keys(sd) == {"cls_token": 1, "blocks.0.attn.qkv.weight": 2, "fc.weight": 6}
