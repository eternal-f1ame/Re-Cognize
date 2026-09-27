"""Integration tests for MemoryEnhancedReID model (TransReID backbone)."""

import pytest
import torch
import torch.nn.functional as F

from memory_block.models.model import MemoryConfig, MemoryEnhancedReID
from recognize.backbones import BACKBONE_REGISTRY

# Every test here builds the TransReID backbone, which loads its released weights.
pytestmark = pytest.mark.skipif(
    not BACKBONE_REGISTRY["transreid"].weights_path().exists(),
    reason="needs the released TransReID weights: python scripts/fetch_weights.py transreid")


def _make_config(**overrides):
    """Create a MemoryConfig with sensible test defaults."""
    defaults = dict(
        num_classes=10,
        feat_dim=768,
        use_working_memory=True,
        use_episodic_memory=True,
        episodic_id_drop_rate=0.5,
        backbone_type="transreid",
        freeze_backbone=True,
        image_height=256,
        image_width=128,
    )
    defaults.update(overrides)
    return MemoryConfig(**defaults)


# ───────────────────────────────────────────────────────────────────
# Config defaults
# ───────────────────────────────────────────────────────────────────


class TestMemoryConfig:
    def test_default_loss_weight_rebalance(self):
        """MemoryConfig's default episodic ID-drop rate is 0.5."""
        config = _make_config()
        assert config.episodic_id_drop_rate == 0.5

    def test_lora_defaults(self):
        config = _make_config()
        assert config.use_lora is False
        assert config.lora_rank == 8
        assert config.lora_layers == 4

    def test_the_residual_is_uncapped_unless_asked(self):
        assert _make_config().residual_max_ratio is None

    def test_the_cap_reaches_the_fusion_module(self):
        model = MemoryEnhancedReID(_make_config(residual_max_ratio=0.1))
        assert model.memory_block.fusion.residual_max_ratio == 0.1

    def test_a_checkpoint_config_written_before_the_cap_existed_still_loads(self):
        # a checkpoint config dict without the key must still build, with the field at its default
        old_cfg = {"num_classes": 10, "feat_dim": 768, "backbone_type": "transreid"}
        cfg = MemoryConfig(**{k: v for k, v in old_cfg.items() if k in MemoryConfig.__dataclass_fields__})
        assert cfg.residual_max_ratio is None


# ───────────────────────────────────────────────────────────────────
# Model creation
# ───────────────────────────────────────────────────────────────────


class TestModelCreation:
    def test_create_with_memory(self):
        config = _make_config()
        model = MemoryEnhancedReID(config)
        assert model.memory_block is not None
        assert model.memory_block.episodic_id_drop_rate == 0.5

    def test_create_without_memory(self):
        config = _make_config(use_working_memory=False, use_episodic_memory=False)
        model = MemoryEnhancedReID(config)
        assert model.memory_block is None

    def test_backbone_is_frozen(self):
        config = _make_config()
        model = MemoryEnhancedReID(config)
        for p in model.backbone_wrapper.backbone.parameters():
            assert p.requires_grad is False

    def test_memory_params_are_trainable(self):
        config = _make_config()
        model = MemoryEnhancedReID(config)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0
        # At least memory block params should be trainable
        memory_trainable = [n for n in trainable if "memory_block" in n]
        assert len(memory_trainable) > 0


# ───────────────────────────────────────────────────────────────────
# LoRA integration
# ───────────────────────────────────────────────────────────────────


class TestLoRAIntegration:
    def test_lora_creates_trainable_backbone_params(self):
        config = _make_config(use_lora=True, lora_rank=4, lora_layers=2)
        model = MemoryEnhancedReID(config)
        # Should have some LoRA params in backbone
        from memory_block.models.lora import get_lora_parameters
        lora_params = get_lora_parameters(model)
        assert len(lora_params) > 0

    def test_lora_off_no_extra_params(self):
        config = _make_config(use_lora=False)
        model = MemoryEnhancedReID(config)
        from memory_block.models.lora import get_lora_parameters
        lora_params = get_lora_parameters(model)
        assert len(lora_params) == 0

    def test_train_mode_keeps_backbone_eval_but_lora_train(self):
        config = _make_config(use_lora=True, lora_rank=4, lora_layers=2)
        model = MemoryEnhancedReID(config)
        model.train()
        # Backbone overall should be eval
        assert not model.backbone_wrapper.backbone.training
        # But LoRA modules should be in train mode
        from memory_block.models.lora import LoRALinear
        for m in model.backbone_wrapper.backbone.modules():
            if isinstance(m, LoRALinear):
                assert m.training, "LoRA modules should be in train mode"


# ───────────────────────────────────────────────────────────────────
# Forward pass
# ───────────────────────────────────────────────────────────────────


class TestForwardPass:
    @pytest.fixture
    def model(self):
        config = _make_config()
        m = MemoryEnhancedReID(config)
        # Initialize episodic memory so forward works
        m.memory_block.episodic_memory.prototypes.data = torch.randn(
            10, config.slots_per_char, config.feat_dim
        )
        m.memory_block.episodic_memory.slot_filled.fill_(True)
        m.memory_block.episodic_memory.char_initialized.fill_(True)
        # char_initialized already set via fill_(True) above
        return m

    def test_train_forward_returns_all_keys(self, model):
        model.train()
        x = torch.randn(4, 3, 256, 128)
        labels = torch.randint(0, 10, (4,))
        out = model(x, char_ids=labels, use_memory=True, return_all=True)
        assert "logits" in out
        assert "bn_feat" in out
        assert "final_feat" in out

    def test_eval_forward_without_char_ids(self, model):
        model.eval()
        x = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            out = model(x, char_ids=None, use_memory=True, return_all=True)
        assert "final_feat" in out
        assert out["final_feat"].shape == (2, 768)

    def test_no_memory_forward(self):
        config = _make_config(use_working_memory=False, use_episodic_memory=False)
        model = MemoryEnhancedReID(config)
        model.eval()
        x = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            out = model(x, use_memory=False, return_all=True)
        assert "bn_feat" in out
        assert out["bn_feat"].shape[1] == 768


# ───────────────────────────────────────────────────────────────────
# Checkpoint round-trip
# ───────────────────────────────────────────────────────────────────


class TestCheckpointRoundTrip:
    def test_save_load_with_lora(self, tmp_path):
        """Save a model with LoRA, reload, verify identical output."""
        config = _make_config(use_lora=True, lora_rank=4, lora_layers=2)
        model1 = MemoryEnhancedReID(config)
        model1.eval()

        # Set some nonzero LoRA weights
        from memory_block.models.lora import LoRALinear
        for m in model1.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data.normal_(0, 0.01)

        # Save
        checkpoint = {
            "model_state_dict": model1.state_dict(),
            "config": vars(config) if hasattr(config, "__dict__") else config.__dict__,
        }
        path = tmp_path / "test_ckpt.pth"
        torch.save(checkpoint, path)

        # Load into fresh model
        model2 = MemoryEnhancedReID(config)
        loaded = torch.load(path, weights_only=False)
        model2.load_state_dict(loaded["model_state_dict"])
        model2.eval()

        # Compare outputs
        x = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            out1 = model1(x, use_memory=False, return_all=True)
            out2 = model2(x, use_memory=False, return_all=True)
        assert torch.allclose(out1["bn_feat"], out2["bn_feat"], atol=1e-5)
