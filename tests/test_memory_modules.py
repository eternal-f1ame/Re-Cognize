"""Tests for memory modules: WorkingMemory, EpisodicMemory, GatedFusion, MemoryBlock."""

import pytest
import torch
import torch.nn.functional as F

from memory_block.models.memory_modules import (
    WorkingMemory,
    EpisodicMemory,
    GatedFusion,
    MemoryBlock,
)


# ───────────────────────────────────────────────────────────────────
# WorkingMemory
# ───────────────────────────────────────────────────────────────────


class TestWorkingMemory:
    @pytest.fixture
    def wm(self, num_characters, feat_dim):
        return WorkingMemory(
            num_characters=num_characters, capacity=4, feat_dim=feat_dim
        )

    def test_output_shape(self, wm, random_features, random_labels):
        out = wm(random_features, char_ids=random_labels)
        assert out.shape == random_features.shape

    def test_no_char_ids_returns_projection(self, wm, random_features):
        """When char_ids is None, working memory should skip and project input."""
        out = wm(random_features, char_ids=None)
        assert out.shape == random_features.shape

    def test_buffer_update(self, wm, feat_dim):
        """After update, buffer should contain the stored feature."""
        feat = torch.randn(1, feat_dim)
        char_id = torch.tensor([0])
        wm(feat, char_ids=char_id, update_memory=True)
        # Buffer for char 0 should have at least 1 entry
        assert wm.memory_filled[0].item() >= 1

    def test_reset_clears_buffers(self, wm, feat_dim):
        """Reset should zero all buffers."""
        feat = torch.randn(1, feat_dim)
        wm(feat, char_ids=torch.tensor([0]), update_memory=True)
        wm.reset()
        assert wm.memory_filled[0].item() == 0

    def test_fifo_capacity(self, wm, feat_dim):
        """Buffer should not exceed capacity."""
        for i in range(10):  # capacity is 4
            feat = torch.randn(1, feat_dim)
            wm(feat, char_ids=torch.tensor([0]), update_memory=True)
        assert wm.memory_filled[0].item() <= wm.capacity

    def test_fifo_eviction(self, wm, feat_dim):
        """FIFO buffer should evict the oldest entry when capacity is reached."""
        for i in range(5):  # capacity is 4, so 0 is evicted.
            feat = torch.ones(1, feat_dim) * i
            wm(feat, char_ids=torch.tensor([0]), update_memory=True)
            
        assert wm.memory_filled[0].item() == 4
        assert not (wm.memory_bank[0] == 0).all(dim=-1).any()


# ───────────────────────────────────────────────────────────────────
# EpisodicMemory
# ───────────────────────────────────────────────────────────────────


class TestEpisodicMemory:
    @pytest.fixture
    def em(self, num_characters, feat_dim):
        return EpisodicMemory(
            num_characters=num_characters, slots_per_char=3, feat_dim=feat_dim
        )

    def test_output_shape_with_char_ids(self, em, random_features, random_labels):
        """Known-ID mode should return (B, D) features."""
        # Initialize prototypes first
        em.prototypes.data = torch.randn_like(em.prototypes)
        em.slot_filled.fill_(True)
        em.char_initialized.fill_(True)
        # num_stored_chars tracked via char_initialized
        out = em(random_features, char_ids=random_labels)
        assert out.shape == random_features.shape

    def test_output_shape_without_char_ids(self, em, random_features):
        """Search-all mode should return (B, D) features."""
        em.prototypes.data = torch.randn_like(em.prototypes)
        em.slot_filled.fill_(True)
        em.char_initialized.fill_(True)
        # num_stored_chars tracked via char_initialized
        out = em(random_features, char_ids=None)
        assert out.shape == random_features.shape

    def test_initialize_from_support(self, em, feat_dim):
        """initialize_from_support should populate prototypes."""
        # 3 characters, 5 features each
        features = torch.randn(15, feat_dim)
        char_ids = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
        em.initialize_from_support(features, char_ids, method="diverse")
        assert em.char_initialized[0].item() is True
        assert em.char_initialized[1].item() is True
        assert em.char_initialized[2].item() is True
        assert em.get_num_initialized_characters() == 3

    def test_reset_clears_prototypes(self, em, feat_dim):
        """Reset should clear all prototype state."""
        features = torch.randn(5, feat_dim)
        char_ids = torch.tensor([0] * 5)
        em.initialize_from_support(features, char_ids)
        em.reset()
        assert em.get_num_initialized_characters() == 0
        assert not em.char_initialized.any()

    def test_update_fills_slots(self, em, feat_dim):
        """Single-feature update should fill a slot."""
        em.char_initialized[0] = True
        # char_initialized[0] already set above
        feat = F.normalize(torch.randn(1, feat_dim), dim=-1)
        em.update(feat, torch.tensor([0]))
        assert em.slot_filled[0, 0].item() is True

    def test_empty_memory_returns_projection(self, em, feat_dim):
        """Querying empty memory (char_ids=None) should return projected input."""
        feat = torch.randn(2, feat_dim)
        out = em(feat, char_ids=None)
        assert out.shape == feat.shape

    def test_slot_rollover(self, em, feat_dim):
        """When episodic memory slots are full, it should roll over back to index 0."""
        em.char_initialized[0] = True
        em.training = False # simulate test-time query mode to prevent drops
        
        torch.manual_seed(123)
        # capacity is 3 slots
        last_feat = None
        for i in range(4):
            # i=0 will be replaced by i=3. We need distinct vectors
            feat = F.normalize(torch.randn(1, feat_dim) + i, dim=-1)
            em.update(feat, torch.tensor([0]))
            if i == 3:
                last_feat = feat
            
        assert em.slot_filled[0].all()  # All 3 slots should be filled
        
        # The last entry (i=3) must have replaced the most similar slot.
        # Let's assert that last_feat is now in the prototypes
        sims = F.cosine_similarity(last_feat, em.prototypes[0], dim=-1)
        assert sims.max().item() > 0.999


# ───────────────────────────────────────────────────────────────────
# GatedFusion
# ───────────────────────────────────────────────────────────────────


class TestGatedFusion:
    @pytest.fixture
    def fusion(self, feat_dim):
        return GatedFusion(feat_dim=feat_dim)

    def test_output_shape(self, fusion, random_features):
        working = torch.randn_like(random_features)
        episodic = torch.randn_like(random_features)
        out, gate = fusion(random_features, working, episodic)
        assert out.shape == random_features.shape
        assert gate.shape == (random_features.shape[0], 2)

    def test_gate_sums_to_one(self, fusion, random_features):
        """Gate weights should sum to 1 (softmax)."""
        working = torch.randn_like(random_features)
        episodic = torch.randn_like(random_features)
        _, gate = fusion(random_features, working, episodic)
        sums = gate.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestResidualCap:
    """The residual may be capped at a fraction of the feature norm (`residual_max_ratio`, off by default)."""

    def test_uncapped_by_default(self, feat_dim):
        assert GatedFusion(feat_dim=feat_dim).residual_max_ratio is None

    def test_a_small_residual_passes_through_untouched(self, feat_dim):
        fusion = GatedFusion(feat_dim=feat_dim, residual_max_ratio=0.1)
        feat = torch.randn(4, feat_dim)
        delta = torch.randn(4, feat_dim) * 1e-4          # the small-init regime at the start of training
        assert torch.equal(fusion.clip_residual(delta, feat), delta)

    def test_a_large_residual_is_projected_onto_the_ball(self, feat_dim):
        fusion = GatedFusion(feat_dim=feat_dim, residual_max_ratio=0.1)
        feat = torch.randn(4, feat_dim)
        delta = torch.randn(4, feat_dim) * 10
        out = fusion.clip_residual(delta, feat)
        limit = 0.1 * feat.norm(dim=-1)
        assert torch.allclose(out.norm(dim=-1), limit, rtol=1e-5)
        # direction is preserved: only the magnitude is touched
        cos = torch.nn.functional.cosine_similarity(out, delta, dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)

    def test_the_cap_bounds_how_far_the_output_moves(self, feat_dim):
        torch.manual_seed(0)
        fusion = GatedFusion(feat_dim=feat_dim, residual_max_ratio=0.05)
        with torch.no_grad():                            # undo the small init: force a large residual
            fusion.output_proj[-1].weight.mul_(1000.0)
        feat = torch.randn(8, feat_dim)
        out, _ = fusion(feat, torch.randn(8, feat_dim), torch.randn(8, feat_dim))
        moved = (out - feat).norm(dim=-1) / feat.norm(dim=-1)
        assert (moved <= 0.05 + 1e-5).all() and (moved > 0.04).any()

    def test_the_direction_still_learns_when_the_cap_binds(self, feat_dim):
        fusion = GatedFusion(feat_dim=feat_dim, residual_max_ratio=0.1)
        feat = torch.randn(2, feat_dim)
        delta = (torch.randn(2, feat_dim) * 10).requires_grad_(True)
        fusion.clip_residual(delta, feat).sum().backward()
        assert delta.grad is not None and delta.grad.abs().sum() > 0

    def test_the_block_passes_the_cap_to_its_fusion(self, num_characters, feat_dim):
        block = MemoryBlock(num_characters=num_characters, feat_dim=feat_dim, residual_max_ratio=0.2)
        assert block.fusion.residual_max_ratio == 0.2

# ───────────────────────────────────────────────────────────────────
# MemoryBlock
# ───────────────────────────────────────────────────────────────────


class TestMemoryBlock:
    @pytest.fixture
    def block(self, num_characters, feat_dim):
        return MemoryBlock(
            num_characters=num_characters,
            working_capacity=4,
            slots_per_char=3,
            feat_dim=feat_dim,
            episodic_id_drop_rate=0.5,
        )

    def test_forward_with_char_ids(self, block, random_features, random_labels, feat_dim):
        """Forward with char_ids should produce dict with expected keys."""
        block.episodic_memory.prototypes.data = torch.randn_like(block.episodic_memory.prototypes)
        block.episodic_memory.slot_filled.fill_(True)
        block.episodic_memory.char_initialized.fill_(True)
        # char_initialized already set via fill_(True) above
        result = block(random_features, char_ids=random_labels)
        assert set(result.keys()) == {"output", "working_out", "episodic_out", "gate_weights"}
        assert result["output"].shape == random_features.shape
        assert result["gate_weights"].shape == (random_features.shape[0], 2)

    def test_forward_without_char_ids(self, block, random_features, feat_dim):
        """Forward without char_ids (test-time mode) should also work."""
        block.episodic_memory.prototypes.data = torch.randn_like(block.episodic_memory.prototypes)
        block.episodic_memory.slot_filled.fill_(True)
        block.episodic_memory.char_initialized.fill_(True)
        # char_initialized already set via fill_(True) above
        result = block(random_features, char_ids=None)
        assert result["output"].shape == random_features.shape

    def test_episodic_id_drop_rate_in_training(self, block, random_features, random_labels):
        """In training mode with episodic_id_drop_rate=1.0, char_ids should always be dropped."""
        block.episodic_id_drop_rate = 1.0
        block.train()
        block.episodic_memory.prototypes.data = torch.randn_like(block.episodic_memory.prototypes)
        block.episodic_memory.slot_filled.fill_(True)
        block.episodic_memory.char_initialized.fill_(True)
        # char_initialized already set via fill_(True) above
        # Should not error even though it always drops char_ids
        result = block(random_features, char_ids=random_labels)
        assert result["output"].shape == random_features.shape

    def test_episodic_id_drop_rate_zero_no_drop(self, block, random_features, random_labels):
        """With drop_rate=0, char_ids should always be passed through."""
        block.episodic_id_drop_rate = 0.0
        block.train()
        block.episodic_memory.prototypes.data = torch.randn_like(block.episodic_memory.prototypes)
        block.episodic_memory.slot_filled.fill_(True)
        block.episodic_memory.char_initialized.fill_(True)
        # char_initialized already set via fill_(True) above
        result = block(random_features, char_ids=random_labels)
        assert result["output"].shape == random_features.shape

    def test_eval_mode_never_drops_ids(self, block, random_features, random_labels):
        """In eval mode, char_ids should never be dropped regardless of drop_rate."""
        block.episodic_id_drop_rate = 1.0  # Would always drop in train mode
        block.eval()
        block.episodic_memory.prototypes.data = torch.randn_like(block.episodic_memory.prototypes)
        block.episodic_memory.slot_filled.fill_(True)
        block.episodic_memory.char_initialized.fill_(True)
        # char_initialized already set via fill_(True) above
        # Should use known-ID mode (not search-all) because eval mode skips drop
        result = block(random_features, char_ids=random_labels)
        assert result["output"].shape == random_features.shape

    def test_reset(self, block, random_features, random_labels, feat_dim):
        """Reset should clear all memory state."""
        block.episodic_memory.initialize_from_support(
            random_features, random_labels, method="diverse"
        )
        block.working_memory(random_features[:1], char_ids=random_labels[:1], update_memory=True)
        block.reset()
        assert block.episodic_memory.get_num_initialized_characters() == 0
        assert block.working_memory.memory_filled[0].item() == 0
