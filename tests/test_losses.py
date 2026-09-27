"""Tests for loss functions: individual losses and CombinedMemoryLoss."""

import pytest
import torch
import torch.nn.functional as F

from memory_block.training.losses import (
    BatchHardTripletLoss,
    MemoryConsistencyLoss,
    PrototypeClassificationLoss,
    CombinedMemoryLoss,
)


# ───────────────────────────────────────────────────────────────────
# BatchHardTripletLoss
# ───────────────────────────────────────────────────────────────────


class TestBatchHardTripletLoss:
    @pytest.fixture
    def triplet_loss(self):
        return BatchHardTripletLoss(margin=0.3)

    def test_output_is_scalar(self, triplet_loss, random_features, random_labels):
        loss, info = triplet_loss(random_features, random_labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_identical_embeddings_returns_margin(self, triplet_loss):
        """All same embeddings -> distance=0 for all pairs -> loss clamps to margin."""
        feats = torch.ones(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss, _ = triplet_loss(feats, labels)
        # With margin=0.3, max(margin + 0 - 0, 0) = 0.3
        assert abs(loss.item() - 0.3) < 1e-5

    def test_gradient_flows(self, triplet_loss, random_features):
        # Use fixed labels that guarantee positive pairs so loss > 0 and grad flows
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        feats = random_features.clone().requires_grad_(True)
        loss, _ = triplet_loss(feats, labels)
        loss.backward()
        assert feats.grad is not None



# ───────────────────────────────────────────────────────────────────
# MemoryConsistencyLoss
# ───────────────────────────────────────────────────────────────────


class TestMemoryConsistencyLoss:
    @pytest.fixture
    def mem_loss(self):
        return MemoryConsistencyLoss()

    def test_output_is_scalar(self, mem_loss, random_features, random_labels):
        raw_feat = random_features
        final_feat = random_features + torch.randn_like(random_features) * 0.1
        loss = mem_loss(final_feat, raw_feat, random_labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_same_identity_features_lower_loss(self, mem_loss, feat_dim):
        """Features from same identity should produce lower loss than random."""
        torch.manual_seed(42)
        
        # We must use a mix of positive and negative labels for InfoNCE to work
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        
        # For "close" features, make items 0/1 close to each other, and 2/3 close to each other
        base_0 = F.normalize(torch.randn(1, feat_dim), dim=-1)
        base_1 = F.normalize(torch.randn(1, feat_dim), dim=-1)
        close_0 = F.normalize(base_0.expand(2, -1) + torch.randn(2, feat_dim) * 0.01, dim=-1)
        close_1 = F.normalize(base_1.expand(2, -1) + torch.randn(2, feat_dim) * 0.01, dim=-1)
        close_feats = torch.cat([close_0, close_1], dim=0)
        
        loss_close = mem_loss(close_feats, close_feats.clone(), labels)
        
        # For random features, everything is uncorrelated
        rand_feats = F.normalize(torch.randn(4, feat_dim), dim=-1)
        loss_rand = mem_loss(rand_feats, F.normalize(torch.randn(4, feat_dim), dim=-1), labels)
        
        assert loss_close.item() >= 0
        assert loss_close.item() < loss_rand.item()


    def test_self_pair_makes_singleton_batches_informative(self):
        """One crop per identity: no cross-sample positive exists, only the self pair."""
        torch.manual_seed(0)
        labels = torch.tensor([0, 1, 2, 3])
        feats = F.normalize(torch.randn(4, 8), dim=-1)
        with_self = MemoryConsistencyLoss(include_self=True)(feats, feats.clone(), labels)
        without_self = MemoryConsistencyLoss(include_self=False)(feats, feats.clone(), labels)
        assert with_self.item() > 0 and torch.isfinite(with_self)
        assert without_self.item() == 0.0                        # without the self pair: nothing to learn from

    def test_identical_features_beat_perturbed_ones(self):
        torch.manual_seed(1)
        labels = torch.tensor([0, 0, 1, 1])
        base = F.normalize(torch.randn(4, 8), dim=-1)
        loss = MemoryConsistencyLoss()
        exact = loss(base, base.clone(), labels).item()
        for scale in (0.2, 0.4, 0.6, 0.8, 1.0):
            perturbed = F.normalize(base + torch.randn_like(base) * scale, dim=-1)
            assert loss(perturbed, base.clone(), labels).item() > exact

    def test_gradient_reaches_the_memory_output(self):
        torch.manual_seed(2)
        labels = torch.tensor([0, 0, 1, 1])
        mem = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_(True)
        orig = F.normalize(torch.randn(4, 8), dim=-1)
        MemoryConsistencyLoss()(mem, orig, labels).backward()
        assert mem.grad is not None and mem.grad.abs().sum() > 0


class TestMemoryTermIsGated:
    """The memory-consistency term applies only to models that ran a memory block."""

    def _output(self, with_gate: bool, B=4, D=8):
        torch.manual_seed(0)
        feat = F.normalize(torch.randn(B, D), dim=-1)
        out = {"bn_feat": feat, "final_feat": feat, "pre_memory_feat": feat.clone(),
               "logits": torch.randn(B, 3)}
        if with_gate:
            out["gate_weights"] = torch.rand(B, 2)
        return out

    def test_no_memory_path_means_no_memory_loss(self):
        crit = CombinedMemoryLoss(num_classes=3, feat_dim=8, ce_weight=0.0)
        labels = torch.tensor([0, 0, 1, 1])
        _, no_gate = crit(self._output(False), labels)
        _, with_gate = crit(self._output(True), labels)
        assert "memory_loss" not in no_gate and "memory_loss" in with_gate

    def test_zero_weight_disables_the_term(self):
        crit = CombinedMemoryLoss(num_classes=3, feat_dim=8, ce_weight=0.0, memory_weight=0.0)
        _, d = crit(self._output(True), torch.tensor([0, 0, 1, 1]))
        assert "memory_loss" not in d


# ───────────────────────────────────────────────────────────────────
# PrototypeClassificationLoss
# ───────────────────────────────────────────────────────────────────


class TestPrototypeClassificationLoss:
    @pytest.fixture
    def proto_loss(self):
        return PrototypeClassificationLoss()

    def test_output_is_scalar(self, proto_loss, feat_dim, num_characters):
        feats = F.normalize(torch.randn(8, feat_dim), dim=-1)
        labels = torch.randint(0, num_characters, (8,))
        prototypes = torch.randn(num_characters, 3, feat_dim)
        mask = torch.ones(num_characters, 3, dtype=torch.bool)
        loss, info = proto_loss(feats, labels, prototypes, mask)
        assert loss.dim() == 0

    def test_gradient_flows(self, proto_loss, feat_dim, num_characters):
        feats = F.normalize(torch.randn(8, feat_dim), dim=-1).requires_grad_(True)
        labels = torch.randint(0, num_characters, (8,))
        prototypes = torch.randn(num_characters, 3, feat_dim)
        mask = torch.ones(num_characters, 3, dtype=torch.bool)
        loss, _ = proto_loss(feats, labels, prototypes, mask)
        loss.backward()
        assert feats.grad is not None



# ───────────────────────────────────────────────────────────────────
# CombinedMemoryLoss
# ───────────────────────────────────────────────────────────────────


class TestCombinedMemoryLoss:
    @pytest.fixture
    def combined_loss(self, num_characters, feat_dim):
        return CombinedMemoryLoss(num_classes=num_characters, feat_dim=feat_dim)

    def test_default_weights(self, combined_loss):
        """The default loss weights: CE 0.3, prototype 1.0, triplet 1.0."""
        assert combined_loss.ce_weight == 0.3
        assert combined_loss.prototype_weight == 1.0
        assert combined_loss.triplet_weight == 1.0

    def test_forward_returns_loss_and_dict(self, combined_loss, feat_dim, num_characters):
        B = 8
        model_output = {
            "logits": torch.randn(B, num_characters),
            "bn_feat": F.normalize(torch.randn(B, feat_dim), dim=-1),
            "final_feat": F.normalize(torch.randn(B, feat_dim), dim=-1),
        }
        labels = torch.randint(0, num_characters, (B,))
        loss, loss_dict = combined_loss(model_output, labels)
        assert loss.dim() == 0
        assert "ce_loss" in loss_dict
        assert "triplet_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_prototype_loss_included_when_prototypes_provided(
        self, combined_loss, feat_dim, num_characters
    ):
        B = 8
        model_output = {
            "logits": torch.randn(B, num_characters),
            "bn_feat": F.normalize(torch.randn(B, feat_dim), dim=-1),
            "final_feat": F.normalize(torch.randn(B, feat_dim), dim=-1),
            "prototypes": torch.randn(num_characters, 3, feat_dim),
            "prototype_mask": torch.ones(num_characters, 3, dtype=torch.bool),
        }
        labels = torch.randint(0, num_characters, (B,))
        loss, loss_dict = combined_loss(model_output, labels)
        assert "prototype_loss" in loss_dict

    def test_gate_weights_logged_when_provided(
        self, combined_loss, feat_dim, num_characters
    ):
        B = 8
        model_output = {
            "logits": torch.randn(B, num_characters),
            "bn_feat": F.normalize(torch.randn(B, feat_dim), dim=-1),
            "final_feat": F.normalize(torch.randn(B, feat_dim), dim=-1),
            "gate_weights": torch.softmax(torch.randn(B, 2), dim=-1),
        }
        labels = torch.randint(0, num_characters, (B,))
        loss, loss_dict = combined_loss(model_output, labels)
        assert "gate_working" in loss_dict
        assert "gate_episodic" in loss_dict

    def test_gradient_flows_through_all_losses(
        self, combined_loss, feat_dim, num_characters
    ):
        B = 8
        bn_feat = F.normalize(torch.randn(B, feat_dim), dim=-1).requires_grad_(True)
        model_output = {
            "logits": torch.randn(B, num_characters, requires_grad=True),
            "bn_feat": bn_feat,
            "final_feat": bn_feat,
            "pre_memory_feat": bn_feat,
        }
        labels = torch.randint(0, num_characters, (B,))
        loss, _ = combined_loss(model_output, labels)
        loss.backward()
        assert bn_feat.grad is not None
