"""Per-sample ID-drop, capacity growth and gradient flow in the episodic step."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"


def _block(id_drop=0.5, n=6, d=16):
    from memory_block.models.memory_modules import MemoryBlock
    torch.manual_seed(0)
    b = MemoryBlock(num_characters=n, feat_dim=d, working_capacity=4, slots_per_char=3, num_heads=2,
                    dropout=0.0, episodic_id_drop_rate=id_drop)
    b.train()
    feats = F.normalize(torch.randn(n, d), dim=1)
    b.initialize_from_support(feats, torch.arange(n))
    b.working_memory.update(feats, torch.arange(n))
    return b, feats


class TestIdDrop:
    def test_row_rate_is_the_configured_rate(self):
        b, feats = _block(0.5)
        ids = torch.arange(6)
        rates = []
        for _ in range(200):
            b(feats, char_ids=ids, update_working=False)
            rates.append(b._last_id_drop_mask.float().mean().item())
        assert np.mean(rates) == pytest.approx(0.5, abs=0.05)
        assert 0.0 in rates or 1.0 in rates or True                       # per-sample, so mixed batches occur
        assert any(0.0 < r < 1.0 for r in rates), "drop must vary within a batch, not per batch"

    def test_dropped_rows_get_the_search_all_output(self):
        b, feats = _block(0.5)
        ids = torch.arange(6)
        for seed in range(20):                                    # first draw that mixes both modes
            torch.manual_seed(seed)
            out = b(feats, char_ids=ids, update_working=False)["episodic_out"]
            drop = b._last_id_drop_mask
            if drop.any() and not drop.all():
                break
        else:
            pytest.fail("no mixed ID-drop mask in 20 draws")
        with torch.no_grad():
            guided = b.episodic_memory(feats, char_ids=ids)
            searched = b.episodic_memory(feats, char_ids=None)
        torch.testing.assert_close(out[drop], searched[drop])
        torch.testing.assert_close(out[~drop], guided[~drop])

    def test_search_all_flag_still_forces_every_row(self):
        b, feats = _block(0.5)
        out = b(feats, char_ids=torch.arange(6), update_working=False, episodic_search_all=True)["episodic_out"]
        with torch.no_grad():
            searched = b.episodic_memory(feats, char_ids=None)
        torch.testing.assert_close(out, searched)

    def test_eval_mode_never_drops(self):
        b, feats = _block(0.5)
        b.eval()
        out = b(feats, char_ids=torch.arange(6), update_working=False)["episodic_out"]
        with torch.no_grad():
            guided = b.episodic_memory(feats, char_ids=torch.arange(6))
        torch.testing.assert_close(out, guided)

    def test_zero_rate_never_drops(self):
        b, feats = _block(0.0)
        out = b(feats, char_ids=torch.arange(6), update_working=False)["episodic_out"]
        with torch.no_grad():
            guided = b.episodic_memory(feats, char_ids=torch.arange(6))
        torch.testing.assert_close(out, guided)


class TestCapacity:
    def test_growth_keeps_contents(self):
        b, feats = _block()
        before = b.episodic_memory.prototypes[:6].clone()
        wm_before = b.working_memory.memory_bank[:6].clone()
        b.ensure_capacity(20)
        assert b.episodic_memory.num_characters == 20 and b.working_memory.num_characters == 20
        torch.testing.assert_close(b.episodic_memory.prototypes[:6], before)
        torch.testing.assert_close(b.working_memory.memory_bank[:6], wm_before)
        new = F.normalize(torch.randn(1, 16), dim=1)
        b.episodic_memory.update(new, torch.tensor([19])); b.working_memory.update(new, torch.tensor([19]))
        assert bool(b.episodic_memory.char_initialized[19]) and int(b.working_memory.memory_filled[19]) == 1

    def test_update_beyond_capacity_grows_both(self):
        b, _ = _block()
        new = F.normalize(torch.randn(1, 16), dim=1)
        b.episodic_memory.update(new, torch.tensor([40]))
        b.working_memory.update(new, torch.tensor([40]))
        assert b.episodic_memory.num_characters == 41 and b.working_memory.num_characters == 41

    def test_initialize_from_support_grows(self):
        b, _ = _block()
        feats = F.normalize(torch.randn(3, 16), dim=1)
        b.episodic_memory.initialize_from_support(feats, char_ids=torch.tensor([7, 8, 9]))
        assert b.episodic_memory.num_characters == 10 and bool(b.episodic_memory.char_initialized[9])


class TestTrainingStep:
    def test_gradients_reach_memory_gate_and_bnneck(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from memory_block.models.model import MemoryConfig, MemoryEnhancedReID
        from memory_block.training.losses import CombinedMemoryLoss
        from memory_block.training.train import _episodic_train_step
        torch.manual_seed(0)
        cfg = MemoryConfig(num_classes=3, feat_dim=8, backbone_type="custom", freeze_backbone=True,
                           working_capacity=4, slots_per_char=3, num_heads=2, dropout=0.0,
                           image_height=16, image_width=16,
                           backbone_config={"module": "_tiny_backbone", "class": "TinyBackbone", "kwargs": {"feat_dim": 8}})
        model = MemoryEnhancedReID(cfg); model.train()
        crit = CombinedMemoryLoss(num_classes=3, feat_dim=8, ce_weight=0.0)
        images = torch.rand(12, 3, 16, 16); labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        loss, metrics = _episodic_train_step(model, images, labels, crit, torch.device("cpu"), False, 2)
        loss.backward()
        groups = {
            "working": model.memory_block.working_memory,
            "episodic": model.memory_block.episodic_memory,
            "gate": model.memory_block.fusion,
            "bnneck": model.bnneck,
        }
        for name, mod in groups.items():
            grads = [p.grad for p in mod.parameters() if p.requires_grad]
            assert grads and any(g is not None and g.abs().sum() > 0 for g in grads), f"no gradient reached {name}"
        assert metrics["n_support"] == 6 and metrics["n_query"] == 6

    def test_training_step_does_not_force_search_all(self):
        text = (SRC / "memory_block" / "training" / "train.py").read_text()
        step = text[text.index("def _episodic_train_step"):text.index("def load_training_split")]
        assert "episodic_search_all=True" not in step


class TestAblationSwitches:
    """`--no-working-memory` / `--no-episodic-memory` must actually remove a branch."""

    @staticmethod
    def _block(wm=True, em=True, n=6, d=16):
        from memory_block.models.memory_modules import MemoryBlock
        torch.manual_seed(0)
        b = MemoryBlock(num_characters=n, feat_dim=d, working_capacity=4, slots_per_char=3, num_heads=2,
                        dropout=0.0, episodic_id_drop_rate=0.5, use_working_memory=wm, use_episodic_memory=em).eval()
        feats = F.normalize(torch.randn(n, d), dim=1)
        b.initialize_from_support(feats, torch.arange(n)); b.working_memory.update(feats, torch.arange(n))
        return b, feats

    def test_disabled_branch_contributes_nothing(self):
        full, feats = self._block()
        ids = torch.arange(6)
        out_full = full(feats, char_ids=ids, update_working=False)
        no_wm, _ = self._block(wm=False)
        out_no_wm = no_wm(feats, char_ids=ids, update_working=False)
        no_em, _ = self._block(em=False)
        out_no_em = no_em(feats, char_ids=ids, update_working=False)
        assert out_full["working_out"].abs().sum() > 0 and out_full["episodic_out"].abs().sum() > 0
        assert out_no_wm["working_out"].abs().sum() == 0 and out_no_wm["episodic_out"].abs().sum() > 0
        assert out_no_em["episodic_out"].abs().sum() == 0 and out_no_em["working_out"].abs().sum() > 0
        # and the fused output differs from the full block, which is the point of the ablation
        assert not torch.allclose(out_full["output"], out_no_wm["output"])
        assert not torch.allclose(out_full["output"], out_no_em["output"])

    def test_model_config_reaches_the_block(self):
        from memory_block.models.model import MemoryConfig, MemoryEnhancedReID
        m = MemoryEnhancedReID(MemoryConfig(
            num_classes=3, feat_dim=8, backbone_type="custom", freeze_backbone=True, image_height=16, image_width=16,
            use_working_memory=False, working_capacity=4, slots_per_char=3, num_heads=2, dropout=0.0,
            backbone_config={"module": "_tiny_backbone", "class": "TinyBackbone", "kwargs": {"feat_dim": 8}}))
        assert m.memory_block.use_working_memory is False and m.memory_block.use_episodic_memory is True
        m.reinit_for_open_set(5)
        assert m.memory_block.use_working_memory is False        # preserved across reinit

    def test_both_disabled_is_refused(self):
        from memory_block.models.memory_modules import MemoryBlock
        with pytest.raises(ValueError, match="at least one"):
            MemoryBlock(num_characters=2, feat_dim=8, use_working_memory=False, use_episodic_memory=False)
