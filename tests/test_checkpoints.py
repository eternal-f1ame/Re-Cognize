"""Tests for recognize.checkpoints with stub models."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test_features import StubModel


class _HFWrapper(nn.Module):
    """Stands in for BackboneWrapper: an HF-style backbone config that carries a mask ratio."""

    def __init__(self, lin, mask_ratio=0.75):
        super().__init__()
        self.lin = lin
        self.backbone = SimpleNamespace(config=SimpleNamespace(mask_ratio=mask_ratio))

    def forward(self, x):
        return None, self.lin(x)


class _Wrapped(StubModel):
    """StubModel whose wrapper has ViT-MAE masking switched on."""

    def __init__(self, mask_ratio=0.75, **kw):
        super().__init__(**kw)
        self.backbone_wrapper = _HFWrapper(self.lin, mask_ratio=mask_ratio)


def _write_ckpt(tmp_path, model, training_mode="memory", backbone="magiv2"):
    """A checkpoint laid out as memory_block.training.train writes it."""
    p = tmp_path / "checkpoints" / backbone / training_mode / "seed0" / "final.pth"
    p.parent.mkdir(parents=True)
    cfg = {"backbone_type": backbone, "num_classes": 3, "feat_dim": 4,
           "use_working_memory": True, "use_episodic_memory": True}
    torch.save({"config": cfg, "model_state_dict": model.state_dict(), "training_mode": training_mode,
                "epoch": 200, "args": {"name": f"{backbone}_{training_mode}_seed0", "normalize": "imagenet"}}, p)
    return p


class TestMasking:
    def test_disable_sets_zero_and_returns_old_value(self):
        from recognize.checkpoints import current_mask_ratio, disable_mae_masking
        m = _Wrapped(mask_ratio=0.75)
        assert disable_mae_masking(m) == 0.75
        assert current_mask_ratio(m) == 0.0 and m.backbone_wrapper.backbone.config.mask_ratio == 0.0

    def test_token_shuffle_is_pinned_to_identity(self):
        from recognize.checkpoints import disable_mae_masking
        seen = {}

        class FakeEmb:
            def random_masking(self, sequence, noise=None):
                seen["noise"] = noise
                return sequence, None, None

        m = _Wrapped(mask_ratio=0.75)
        m.backbone_wrapper.backbone.embeddings = FakeEmb()
        disable_mae_masking(m)
        emb = m.backbone_wrapper.backbone.embeddings
        emb.random_masking(torch.zeros(2, 5, 3))
        assert torch.equal(seen["noise"], torch.arange(5, dtype=torch.float32).unsqueeze(0).expand(2, 5))
        disable_mae_masking(m)                                   # idempotent: no double wrapping
        assert emb.random_masking.__name__ == "canonical_random_masking" and emb._canonical_noise is True

    def test_no_hf_config_returns_none(self):
        """A backbone with no HF config (timm ViT, PASS ViT) has nothing to disable."""
        from recognize.checkpoints import disable_mae_masking

        class _PlainWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(2, 2)

        m = StubModel()
        m.backbone_wrapper = _PlainWrapper()
        assert disable_mae_masking(m) is None


class TestLoadCheckpoint:
    def test_loads_strictly_and_sets_flags(self, tmp_path):
        from recognize.checkpoints import load_checkpoint
        src = _Wrapped(mask_ratio=0.75)
        p = _write_ckpt(tmp_path, src, training_mode="memory", backbone="magiv2")
        built = _Wrapped(mask_ratio=0.75)
        with torch.no_grad():
            built.lin.weight.zero_()
        lm = load_checkpoint(p, device="cpu", construct=lambda cfg: built)
        assert lm.model is built and torch.equal(built.lin.weight, src.lin.weight)
        assert lm.backbone == "magiv2" and lm.checkpoint == p and lm.normalize == "imagenet"
        f = lm.flags
        assert f["training_mode"] == "memory" and f["memory"] is True and f["lora"] is False
        assert f["mask_ratio_at_training"] == 0.75 and f["mask_ratio"] == 0.0
        assert f["config_label"] == f["tag"] == "magiv2_memory_seed0" and f["run_epoch"] == 200

    def test_config_reaches_the_builder(self, tmp_path):
        from recognize.checkpoints import load_checkpoint
        p = _write_ckpt(tmp_path, _Wrapped(), training_mode="baseline", backbone="transreid")
        seen = {}

        def construct(cfg):
            seen.update(backbone=cfg.backbone_type, feat_dim=cfg.feat_dim)
            return _Wrapped()

        lm = load_checkpoint(p, construct=construct)
        assert seen == {"backbone": "transreid", "feat_dim": 4} and lm.flags["training_mode"] == "baseline"

    def test_missing_weights_raise(self, tmp_path):
        from recognize.checkpoints import load_checkpoint
        p = _write_ckpt(tmp_path, _Wrapped())
        bigger = _Wrapped(); bigger.extra = nn.Linear(1, 1)
        with pytest.raises(RuntimeError):
            load_checkpoint(p, construct=lambda cfg: bigger)

    def test_rejects_a_file_that_is_not_a_training_checkpoint(self, tmp_path):
        from recognize.checkpoints import load_checkpoint
        p = tmp_path / "weights.pth"
        torch.save({"state_dict": {}}, p)
        with pytest.raises(RuntimeError, match="not a training checkpoint"):
            load_checkpoint(p, construct=lambda cfg: _Wrapped())
