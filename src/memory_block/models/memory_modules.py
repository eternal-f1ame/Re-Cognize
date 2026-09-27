"""
Memory Modules for Manga Character Re-ID

This module implements the dual-memory architecture inspired by VLM2:
- Working Memory: Short-term context from recent panels (sliding window)
- Episodic Memory: Long-term character identity bank (non-parametric prototypes)
- Gated Fusion: Adaptive combination of memory outputs

Key Design Decisions:
- Prototypes are NON-PARAMETRIC (computed from support set at test time)
- Training learns attention mechanisms that generalize to new characters
- Online updating refines prototypes during inference

Usage Modes:
1. Few-shot initialization: Build prototypes from k examples per character
2. Online refinement: Update prototypes as you process queries
3. Combined: Initialize from support, then refine online
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class WorkingMemory(nn.Module):
    """
    Working Memory: Per-character short-term context from recent panels.

    Each character has its own FIFO buffer storing features from their
    recent appearances. Uses cross-attention to query a character's
    own recent context.

    This helps identify characters in challenging situations:
    - Close-up shots (eye only) -> recalls recent full-body view
    - Chibi/deformed versions -> recalls normal appearance
    - Partial occlusion -> recalls complete view

    When char_ids is None (unknown identity at inference), working memory
    is skipped and contributes a zero delta. This means working memory
    only activates when you already know (or have predicted) which
    character you're looking at, i.e., for subsequent appearances after
    initial identification via episodic memory.

    Args:
        num_characters: Number of characters to track
        capacity: Maximum entries per character's buffer (default: 8)
        feat_dim: Feature dimension (default: 768)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> memory = WorkingMemory(num_characters=10, capacity=8, feat_dim=768)
        >>> current_feat = torch.randn(4, 768)
        >>> char_ids = torch.tensor([0, 1, 2, 3])
        >>> attended = memory.query(current_feat, char_ids)
        >>> memory.update(current_feat, char_ids)
    """

    def __init__(
        self,
        num_characters: int,
        capacity: int = 8,
        feat_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_characters = num_characters
        self.capacity = capacity
        self.feat_dim = feat_dim

        # Per-character memory banks
        self.register_buffer(
            "memory_bank",
            torch.zeros(num_characters, capacity, feat_dim),
        )
        self.register_buffer(
            "memory_ptr",
            torch.zeros(num_characters, dtype=torch.long),
        )
        self.register_buffer(
            "memory_filled",
            torch.zeros(num_characters, dtype=torch.long),
        )

        # Cross-attention: current feature queries memory
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm for stability
        self.query_norm = nn.LayerNorm(feat_dim)
        self.memory_norm = nn.LayerNorm(feat_dim)
        self.output_norm = nn.LayerNorm(feat_dim)

        # Output projection: default (kaiming) init so WM gradients flow
        # freely through this layer.  The single small-init bottleneck lives
        # in GatedFusion.output_proj, which still ensures the *residual*
        # starts near-zero while letting cross-attention learn faster.
        self.output_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )

    def ensure_capacity(self, num_characters: int) -> None:
        """Grow the per-character banks to hold `num_characters` identities, keeping their contents.

        P3 can open more clusters than the stream has identities, so `update` grows the banks
        instead of dropping the extra ids.
        """
        if num_characters <= self.num_characters:
            return
        extra = num_characters - self.num_characters
        dev = self.memory_bank.device
        self.memory_bank = torch.cat([self.memory_bank, torch.zeros(extra, self.capacity, self.feat_dim, device=dev)])
        self.memory_ptr = torch.cat([self.memory_ptr, torch.zeros(extra, dtype=torch.long, device=dev)])
        self.memory_filled = torch.cat([self.memory_filled, torch.zeros(extra, dtype=torch.long, device=dev)])
        self.num_characters = num_characters

    def reset(self) -> None:
        """Reset all character memory banks to empty state."""
        self.memory_bank.zero_()
        self.memory_ptr.zero_()
        self.memory_filled.zero_()

    def reset_character(self, char_id: int) -> None:
        """Reset memory for a specific character."""
        self.memory_bank[char_id].zero_()
        self.memory_ptr[char_id] = 0
        self.memory_filled[char_id] = 0

    def get_valid_memory(self, char_id: int) -> Optional[torch.Tensor]:
        """Get only the filled portion of a character's memory."""
        num_filled = min(self.memory_filled[char_id].item(), self.capacity)
        if num_filled == 0:
            return None
        return self.memory_bank[char_id, :num_filled]

    @torch.no_grad()
    def update(
        self,
        new_features: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> None:
        """
        Update per-character memory with new features (FIFO).

        Args:
            new_features: (B, D) features to add
            char_ids: (B,) character IDs for each feature
        """
        if new_features.dim() == 1:
            new_features = new_features.unsqueeze(0)
        if char_ids.dim() == 0:
            char_ids = char_ids.unsqueeze(0)

        B = new_features.shape[0]

        highest = int(char_ids.max().item()) if B else -1
        if highest >= self.num_characters:
            self.ensure_capacity(highest + 1)

        for i in range(B):
            cid = char_ids[i].item()
            if cid < 0:
                continue
            ptr = self.memory_ptr[cid].item()
            self.memory_bank[cid, ptr] = new_features[i].detach()
            self.memory_ptr[cid] = (ptr + 1) % self.capacity
            self.memory_filled[cid] = min(
                self.memory_filled[cid].item() + 1, self.capacity
            )

    def query(
        self,
        current_feat: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Query per-character memory with current features.

        When char_ids is None, working memory is skipped entirely
        (returns a zero delta). This happens at inference when
        the character identity is not yet known.

        Args:
            current_feat: (B, D) current feature to query with
            char_ids: (B,) character IDs. None to skip working memory.
            return_attention: If True, return attention weights

        Returns:
            output: (B, D) attended memory output
            attn_weights: optional attention weights (if return_attention=True)
        """
        B = current_feat.shape[0]

        # No char_ids → WM has nothing to contribute (identity unknown)
        if char_ids is None:
            if return_attention:
                return torch.zeros_like(current_feat), None
            return torch.zeros_like(current_feat)

        # Build per-sample memory tensors
        # We need to handle variable-length memory per character.
        # Pad all to capacity and use attention mask.
        padded_memory = torch.zeros(
            B, self.capacity, self.feat_dim, device=current_feat.device
        )
        attn_mask = torch.ones(
            B, self.capacity, dtype=torch.bool, device=current_feat.device
        )  # True = masked (do NOT attend)

        any_has_memory = False
        for i in range(B):
            cid = char_ids[i].item()
            if cid < 0 or cid >= self.num_characters:
                continue
            valid = self.get_valid_memory(cid)
            if valid is not None:
                n = valid.shape[0]
                padded_memory[i, :n] = valid
                attn_mask[i, :n] = False  # unmask filled slots
                any_has_memory = True

        if not any_has_memory:
            # No character has memory yet, so WM has nothing to contribute
            if return_attention:
                return torch.zeros_like(current_feat), None
            return torch.zeros_like(current_feat)

        # Track which samples have memory (fully-masked rows cause NaN in MHA)
        has_memory = ~attn_mask.all(dim=1)  # (B,) True where at least one slot is unmasked

        # Normalize
        query = self.query_norm(current_feat).unsqueeze(1)  # (B, 1, D)
        memory = self.memory_norm(padded_memory)  # (B, capacity, D)

        # Cross-attention with key_padding_mask
        attended, attn_weights = self.cross_attention(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=attn_mask,
        )

        attended = attended.squeeze(1)  # (B, D)

        # Output projection (no residual; the single residual lives in GatedFusion)
        output = self.output_proj(self.output_norm(attended))

        # Zero out samples that had no memory
        output = output.masked_fill(~has_memory.unsqueeze(1), 0.0)

        if return_attention:
            return output, attn_weights
        return output

    def forward(
        self,
        current_feat: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Query memory and optionally update it.

        Args:
            current_feat: (B, D) current feature
            char_ids: (B,) character IDs. None to skip working memory.
            update_memory: If True, add current feature to memory after query

        Returns:
            output: (B, D) attended memory output
        """
        output = self.query(current_feat, char_ids)

        if update_memory and char_ids is not None:
            self.update(current_feat, char_ids)

        return output


class EpisodicMemory(nn.Module):
    """
    Episodic Memory: Long-term character identity bank with NON-PARAMETRIC prototypes.

    Key Design: Prototypes are NOT learned parameters. Instead:
    1. Initialize from support set (few-shot)
    2. Refine online during inference

    This ensures the attention mechanisms generalize to NEW characters at test time.

    Update Strategy (from VLM2):
    - Replace the MOST SIMILAR slot (maintains diversity)
    - This ensures memory stores diverse views, not redundant copies

    Args:
        num_characters: Maximum number of characters to track
        slots_per_char: Number of prototype slots per character (default: 5)
        feat_dim: Feature dimension (default: 768)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> memory = EpisodicMemory(num_characters=100, slots_per_char=5)
        >>> # Initialize from support set
        >>> support_features = torch.randn(100, 5, 768)  # k=5 per character
        >>> memory.initialize_from_support(support_features)
        >>> # Query
        >>> query_feat = torch.randn(4, 768)
        >>> char_ids = torch.tensor([0, 1, 2, 3])
        >>> attended = memory(query_feat, char_ids, update_memory=True)
    """

    def __init__(
        self,
        num_characters: int,
        slots_per_char: int = 5,
        feat_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_characters = num_characters
        self.slots_per_char = slots_per_char
        self.feat_dim = feat_dim

        # Prototype memory: NON-PARAMETRIC (buffer, not parameter)
        # Shape: (num_characters, slots_per_char, feat_dim)
        self.register_buffer(
            "prototypes",
            torch.zeros(num_characters, slots_per_char, feat_dim)
        )

        # Track which slots have been filled
        self.register_buffer(
            "slot_filled",
            torch.zeros(num_characters, slots_per_char, dtype=torch.bool)
        )

        # Track which characters have been initialized
        self.register_buffer(
            "char_initialized",
            torch.zeros(num_characters, dtype=torch.bool)
        )

        # Cross-attention for querying (THIS IS WHAT WE LEARN)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms (LEARNED)
        self.query_norm = nn.LayerNorm(feat_dim)
        self.memory_norm = nn.LayerNorm(feat_dim)
        self.output_norm = nn.LayerNorm(feat_dim)

        # Output projection (LEARNED): default (kaiming) init so EM
        # gradients flow freely.  The single small-init bottleneck lives
        # in GatedFusion.output_proj, keeping the residual near-zero at init.
        self.output_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )

    def ensure_capacity(self, num_characters: int) -> None:
        """Grow the prototype bank to `num_characters` identities, keeping existing prototypes."""
        if num_characters <= self.num_characters:
            return
        extra = num_characters - self.num_characters
        dev = self.prototypes.device
        self.prototypes = torch.cat([self.prototypes, torch.zeros(extra, self.slots_per_char, self.feat_dim, device=dev)])
        self.slot_filled = torch.cat([self.slot_filled, torch.zeros(extra, self.slots_per_char, dtype=torch.bool, device=dev)])
        self.char_initialized = torch.cat([self.char_initialized, torch.zeros(extra, dtype=torch.bool, device=dev)])
        self.num_characters = num_characters

    def reset(self) -> None:
        """Reset all prototypes and tracking."""
        self.prototypes.zero_()
        self.slot_filled.zero_()
        self.char_initialized.zero_()

    def reset_character(self, char_id: int) -> None:
        """Reset prototypes for a specific character."""
        self.prototypes[char_id].zero_()
        self.slot_filled[char_id].zero_()
        self.char_initialized[char_id] = False

    @torch.no_grad()
    def initialize_from_support(
        self,
        support_features: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        method: str = "diverse",
    ) -> None:
        """
        Initialize prototypes from support set features.

        Args:
            support_features: Features from support set
                - If char_ids is None: (num_chars, k, feat_dim) - all characters
                - If char_ids provided: (N, feat_dim) - features with corresponding char_ids
            char_ids: (N,) character IDs for each feature (optional)
            method: Initialization method
                - "diverse": Select k most diverse samples (farthest point sampling)
                - "mean": Use mean as single prototype, fill rest with samples
                - "first_k": Just use first k samples
        """
        if char_ids is None:
            # support_features is (num_chars, k, feat_dim)
            num_chars, k, D = support_features.shape
            assert D == self.feat_dim, f"Feature dim mismatch: {D} vs {self.feat_dim}"

            for c in range(min(num_chars, self.num_characters)):
                char_feats = support_features[c]  # (k, D)
                self._initialize_character(c, char_feats, method)
        else:
            # support_features is (N, feat_dim), char_ids is (N,)
            unique_chars = char_ids.unique()

            highest = int(unique_chars.max().item()) if len(unique_chars) else -1
            if highest >= self.num_characters:
                self.ensure_capacity(highest + 1)

            for char_id in unique_chars:
                char_id = char_id.item()
                mask = char_ids == char_id
                char_feats = support_features[mask]  # (k_c, D)
                self._initialize_character(char_id, char_feats, method)

    def _initialize_character(
        self,
        char_id: int,
        features: torch.Tensor,
        method: str = "diverse",
    ) -> None:
        """Initialize prototypes for a single character."""
        k = features.shape[0]
        slots_to_fill = min(k, self.slots_per_char)

        if method == "diverse" and k > self.slots_per_char:
            # Farthest point sampling for diversity
            selected_indices = self._farthest_point_sampling(features, self.slots_per_char)
            selected_feats = features[selected_indices]
        elif method == "mean":
            # Mean as first prototype, then diverse samples
            mean_feat = features.mean(dim=0, keepdim=True)
            if k > 1:
                other_indices = self._farthest_point_sampling(features, self.slots_per_char - 1)
                selected_feats = torch.cat([mean_feat, features[other_indices]], dim=0)
            else:
                selected_feats = mean_feat
            slots_to_fill = selected_feats.shape[0]
        else:  # first_k
            selected_feats = features[:slots_to_fill]

        # Fill slots
        self.prototypes[char_id, :slots_to_fill] = selected_feats.detach()
        self.slot_filled[char_id, :slots_to_fill] = True
        self.char_initialized[char_id] = True

    def _farthest_point_sampling(
        self,
        features: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Select diverse samples using farthest point sampling.

        Args:
            features: (N, D) feature matrix
            num_samples: Number of samples to select

        Returns:
            indices: (num_samples,) selected indices
        """
        N = features.shape[0]
        num_samples = min(num_samples, N)

        # Normalize for cosine distance
        features_norm = F.normalize(features, p=2, dim=-1)

        # Start with random point
        selected = [torch.randint(N, (1,)).item()]

        for _ in range(num_samples - 1):
            # Compute distances to all selected points
            selected_feats = features_norm[selected]  # (k, D)
            similarities = torch.mm(features_norm, selected_feats.t())  # (N, k)
            max_similarities, _ = similarities.max(dim=1)  # (N,)

            # Mask already selected
            max_similarities[selected] = float('inf')

            # Select farthest (least similar)
            farthest = max_similarities.argmin().item()
            selected.append(farthest)

        return torch.tensor(selected, device=features.device)

    def get_character_prototypes(
        self,
        char_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get prototypes for specific characters.

        Args:
            char_ids: (B,) character IDs

        Returns:
            prototypes: (B, slots_per_char, feat_dim)
        """
        return self.prototypes[char_ids]

    def get_valid_prototypes(
        self,
        char_id: int,
    ) -> Optional[torch.Tensor]:
        """Get only filled prototypes for a character."""
        filled = self.slot_filled[char_id]
        if not filled.any():
            return None
        return self.prototypes[char_id, filled]

    @torch.no_grad()
    def update(
        self,
        new_features: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> None:
        """
        Update prototypes with new features (online learning).

        Strategy: Replace the most similar slot to maintain diversity.

        Args:
            new_features: (B, D) new features
            char_ids: (B,) character IDs
        """
        B = new_features.shape[0]
        highest = int(char_ids.max().item()) if B else -1
        if highest >= self.num_characters:
            self.ensure_capacity(highest + 1)

        for i in range(B):
            char_id = char_ids[i].item()
            if char_id < 0:
                continue

            feat = new_features[i].detach()

            # Get current prototypes for this character
            filled = self.slot_filled[char_id]

            if not filled.all():
                # Find first empty slot
                empty_idx = (~filled).nonzero(as_tuple=True)[0][0]
                self.prototypes[char_id, empty_idx] = feat
                self.slot_filled[char_id, empty_idx] = True
                self.char_initialized[char_id] = True
            else:
                # All slots filled: replace most similar (maintains diversity)
                char_protos = self.prototypes[char_id]  # (slots, D)
                similarities = F.cosine_similarity(
                    feat.unsqueeze(0),
                    char_protos,
                    dim=-1,
                )
                most_similar_idx = similarities.argmax().item()
                self.prototypes[char_id, most_similar_idx] = feat

    def query(
        self,
        current_feat: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Query episodic memory.

        If char_ids provided: query that character's prototypes (training/known ID)
        If char_ids is None: query all prototypes (inference/search)

        Args:
            current_feat: (B, D) current feature
            char_ids: (B,) character IDs (optional)
            return_attention: If True, return attention weights

        Returns:
            output: (B, D) attended memory output
            attn_weights: attention weights (if return_attention=True)
        """
        B = current_feat.shape[0]

        # Normalize query
        query = self.query_norm(current_feat).unsqueeze(1)  # (B, 1, D)

        if char_ids is not None:
            # Training/known ID mode: query specific character's prototypes
            char_protos = self.get_character_prototypes(char_ids)  # (B, slots, D)

            # Create attention mask for unfilled slots
            filled_mask = self.slot_filled[char_ids]  # (B, slots)
            attn_mask = ~filled_mask  # True where we should NOT attend

            # Track which samples have NO prototypes at all.
            # MHA with a fully-masked key sequence produces NaN; to avoid that
            # we temporarily unmask these rows so MHA attends to zero vectors.
            # We then zero the output below so garbage values never propagate.
            no_protos = ~filled_mask.any(dim=1)  # (B,) True → no filled slots
            if no_protos.any():
                attn_mask[no_protos] = False  # prevent NaN in MHA

            memory = self.memory_norm(char_protos)

            # Expand mask for multihead attention: (B, 1, slots)
            attn_mask = attn_mask.unsqueeze(1)

        else:
            # Inference/search mode: query all initialized prototypes
            # Flatten all prototypes
            all_protos = self.prototypes.view(-1, self.feat_dim)  # (N*slots, D)
            all_filled = self.slot_filled.view(-1)  # (N*slots,)

            if not all_filled.any():
                # No prototypes, so EM has nothing to contribute
                if return_attention:
                    return torch.zeros_like(current_feat), None
                return torch.zeros_like(current_feat)

            # Only use filled slots
            valid_protos = all_protos[all_filled]  # (M, D)
            memory = self.memory_norm(valid_protos).unsqueeze(0)  # (1, M, D)
            memory = memory.expand(B, -1, -1)
            attn_mask = None
            no_protos = None  # inference path: all have valid (global) protos

        # Cross-attention
        attended, attn_weights = self.cross_attention(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=attn_mask.squeeze(1) if attn_mask is not None else None,
        )

        attended = attended.squeeze(1)  # (B, D)

        # Output projection (no residual; the single residual lives in GatedFusion)
        output = self.output_proj(self.output_norm(attended))

        # Zero out samples that had no prototypes.  Without this, those samples
        # would receive an attention output computed over all-zero prototype
        # vectors: a non-zero garbage delta that the model could learn to
        # rely on spuriously (especially early in training).
        if no_protos is not None and no_protos.any():
            output = output.masked_fill(no_protos.unsqueeze(1), 0.0)

        if return_attention:
            return output, attn_weights
        return output

    def forward(
        self,
        current_feat: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> torch.Tensor:
        """
        Query and optionally update episodic memory.

        Args:
            current_feat: (B, D) current feature
            char_ids: (B,) character IDs
            update_memory: If True, update prototypes with current feature

        Returns:
            output: (B, D) attended memory output
        """
        output = self.query(current_feat, char_ids)

        if update_memory and char_ids is not None:
            self.update(current_feat, char_ids)

        return output

    def get_all_prototypes(self) -> torch.Tensor:
        """Get all prototypes as a gallery for inference."""
        return self.prototypes.view(-1, self.feat_dim)

    def compute_similarity(
        self,
        query_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between query and all prototypes.

        Useful for inference: finding the most similar character.

        Args:
            query_feat: (B, D) query features

        Returns:
            similarities: (B, num_characters) max similarity per character
        """
        B = query_feat.shape[0]

        # Normalize
        query_norm = F.normalize(query_feat, p=2, dim=-1)
        proto_norm = F.normalize(
            self.prototypes.view(-1, self.feat_dim), p=2, dim=-1
        )

        # Compute all similarities
        all_sims = torch.mm(query_norm, proto_norm.t())  # (B, N*slots)

        # Reshape and take max per character
        all_sims = all_sims.view(B, self.num_characters, self.slots_per_char)

        # Mask unfilled slots with -inf before max
        filled_mask = self.slot_filled.unsqueeze(0).expand(B, -1, -1)  # (B, N, slots)
        all_sims = all_sims.masked_fill(~filled_mask, float('-inf'))

        max_sims, _ = all_sims.max(dim=-1)  # (B, num_characters)

        # Mask uninitialized characters with -inf so they can NEVER beat initialized
        # ones in argmax, regardless of the actual similarity values. A fill of 0.0 would let
        # an uninitialized character win whenever all real characters have negative cosine
        # similarity (e.g. early in training or with prototypes from another feature space).
        max_sims = max_sims.masked_fill(
            ~self.char_initialized.unsqueeze(0), float('-inf')
        )

        return max_sims

    def get_num_initialized_characters(self) -> int:
        """Return number of characters with at least one prototype."""
        return self.char_initialized.sum().item()


class GatedFusion(nn.Module):
    """
    Gated Fusion: Adaptive combination of working and episodic memory.

    Learns when to rely on each memory type:
    - Same page, consecutive panels -> favor working memory
    - New chapter, character reappears -> favor episodic memory

    Args:
        feat_dim: Feature dimension (default: 768)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        feat_dim: int = 768,
        dropout: float = 0.1,
        residual_max_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        # Cap on how far the residual may move a feature, as a fraction of that feature's norm.
        # None (the default) leaves the residual unconstrained. The cap addresses a measured
        # defect: the memory raises mAP and lowers Rank-1 on every backbone, and the backbone it
        # displaces least (MagiV2, cosine 0.89 to the no-memory feature) is the one it helps most.
        self.residual_max_ratio = residual_max_ratio

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Output projection: this is the ONLY residual path for memory.
        # No LayerNorm here: LN normalizes any non-zero input to unit variance,
        # defeating the small-scale init. Without LN, small deltas in → small out.
        # Small-scale init on last layer so memory starts as near-no-op;
        # gradients still flow (unlike zero-init which blocks them).
        self.output_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        nn.init.normal_(self.output_proj[-1].weight, std=0.001)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        original_feat: torch.Tensor,
        working_out: torch.Tensor,
        episodic_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse working and episodic memory deltas with single residual.

        Args:
            original_feat: (B, D) original feature (before memory)
            working_out: (B, D) working memory delta (zero when inactive)
            episodic_out: (B, D) episodic memory delta (zero when inactive)

        Returns:
            fused: (B, D) original_feat + projected(gated memory deltas)
            gate_weights: (B, 2) gate weights [working, episodic]
        """
        # Concatenate all features for gate decision
        combined = torch.cat([original_feat, working_out, episodic_out], dim=-1)

        # Compute gate weights
        gate_weights = self.gate(combined)  # (B, 2)

        # Weighted combination
        fused = (
            gate_weights[:, 0:1] * working_out +
            gate_weights[:, 1:2] * episodic_out
        )

        # Output projection with residual
        output = original_feat + self.clip_residual(self.output_proj(fused), original_feat)

        return output, gate_weights

    def clip_residual(self, delta: torch.Tensor, original_feat: torch.Tensor) -> torch.Tensor:
        """Project the residual onto a ball of radius `residual_max_ratio * ||original_feat||`.

        A no-op while the residual is small, so the small-init behaviour at the start of training is
        unchanged; it binds only once the memory tries to move a feature further than the budget.
        The direction keeps its gradient and the magnitude saturates, the same shape as gradient-norm
        clipping; unlike a hard `min`, it stays differentiable in the direction.
        """
        if self.residual_max_ratio is None:
            return delta
        limit = self.residual_max_ratio * original_feat.norm(dim=-1, keepdim=True)
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return delta * torch.clamp(limit / norm, max=1.0)


class MemoryBlock(nn.Module):
    """
    Complete Memory Block combining Working and Episodic Memory.

    This is the main memory module that integrates:
    - Working Memory (panel context) - runtime buffer
    - Episodic Memory (character identity) - non-parametric prototypes
    - Gated Fusion (adaptive combination) - learned

    Training learns:
    - Cross-attention mechanisms (how to query memory)
    - Gated fusion (when to use which memory)
    - Output projections

    Prototypes are NOT learned - they're computed at test time from:
    - Support set (few-shot initialization)
    - Online updates (refinement during inference)

    Args:
        num_characters: Number of characters
        working_capacity: Working memory capacity (default: 8)
        slots_per_char: Episodic memory slots per character (default: 5)
        feat_dim: Feature dimension (default: 768)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        episodic_id_drop_rate: During training, probability of dropping char_ids
            for episodic memory query, forcing it to use the search-all mode
            that matches test-time behavior. (default: 0.5)
    """

    def __init__(
        self,
        num_characters: int,
        working_capacity: int = 8,
        slots_per_char: int = 5,
        feat_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        episodic_id_drop_rate: float = 0.5,
        use_working_memory: bool = True,
        use_episodic_memory: bool = True,
        residual_max_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        if not (use_working_memory or use_episodic_memory):
            raise ValueError("a MemoryBlock needs at least one of working or episodic memory")
        self.num_characters = num_characters
        self.slots_per_char = slots_per_char
        self.feat_dim = feat_dim
        self.episodic_id_drop_rate = episodic_id_drop_rate
        # Ablation switches. They act here in the block, not only in the config, so the "no WM"
        # and "no EM" ablations really train without that branch.
        self.use_working_memory = use_working_memory
        self.use_episodic_memory = use_episodic_memory

        self.working_memory = WorkingMemory(
            num_characters=num_characters,
            capacity=working_capacity,
            feat_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.episodic_memory = EpisodicMemory(
            num_characters=num_characters,
            slots_per_char=slots_per_char,
            feat_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fusion = GatedFusion(
            feat_dim=feat_dim,
            dropout=dropout,
            residual_max_ratio=residual_max_ratio,
        )

    def ensure_capacity(self, num_characters: int) -> None:
        """Grow both memories to hold `num_characters` identities/clusters."""
        self.working_memory.ensure_capacity(num_characters)
        self.episodic_memory.ensure_capacity(num_characters)
        self.num_characters = max(self.num_characters, num_characters)

    def reset(self) -> None:
        """Reset all memory state."""
        self.working_memory.reset()
        self.episodic_memory.reset()

    def reset_working_memory(self) -> None:
        """Reset working memory (e.g., at start of new page/chapter)."""
        self.working_memory.reset()

    def initialize_from_support(
        self,
        support_features: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        method: str = "diverse",
    ) -> None:
        """
        Initialize episodic memory from support set.

        Args:
            support_features: Support set features
            char_ids: Character IDs (optional)
            method: "diverse", "mean", or "first_k"
        """
        self.episodic_memory.initialize_from_support(
            support_features, char_ids, method
        )

    def forward(
        self,
        current_feat: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        update_working: bool = True,
        update_episodic: bool = False,
        episodic_search_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete memory block.

        Args:
            current_feat: (B, D) current feature
            char_ids: (B,) character IDs (optional; None = unknown identity)
            update_working: If True, update working memory
            update_episodic: If True, update episodic memory (online learning)
            episodic_search_all: If True, force episodic memory into search-all
                mode (char_ids=None) regardless of what char_ids was passed.
                Used by `identify_character` in pass 2 at test time; the training
                step leaves it False so that ID-drop decides per sample.

        Returns:
            dict with:
                - output: (B, D) final fused output
                - working_out: (B, D) working memory output
                - episodic_out: (B, D) episodic memory output
                - gate_weights: (B, 2) fusion gate weights
        """
        # Query working memory (per-character; skipped when char_ids is None)
        if self.use_working_memory:
            working_out = self.working_memory(
                current_feat, char_ids=char_ids, update_memory=update_working
            )
        else:
            working_out = torch.zeros_like(current_feat)          # ablation: no working-memory branch

        # Episodic memory ID-drop: during training each sample independently loses its
        # identity signal with probability `episodic_id_drop_rate`, so the cross-attention
        # learns both the identity-guided and the search-all mode it meets at test time.
        # Working memory keeps the real char_ids (it needs them for routing). The drop only
        # happens when the caller does not force `episodic_search_all=True`.
        if not self.use_episodic_memory:
            episodic_out = torch.zeros_like(current_feat)         # ablation: no episodic branch
        elif episodic_search_all or char_ids is None:
            episodic_out = self.episodic_memory(current_feat, char_ids=None, update_memory=False)
            if update_episodic and char_ids is not None:
                self.episodic_memory.update(current_feat, char_ids)
        elif self.training and self.episodic_id_drop_rate > 0:
            drop = torch.rand(current_feat.shape[0], device=current_feat.device) < self.episodic_id_drop_rate
            guided = self.episodic_memory(current_feat, char_ids=char_ids, update_memory=False)
            if drop.all():
                episodic_out = self.episodic_memory(current_feat, char_ids=None, update_memory=False)
            elif not drop.any():
                episodic_out = guided
            else:
                searched = self.episodic_memory(current_feat, char_ids=None, update_memory=False)
                episodic_out = torch.where(drop.unsqueeze(1), searched, guided)
            self._last_id_drop_mask = drop.detach()
            if update_episodic:
                self.episodic_memory.update(current_feat, char_ids)
        else:
            episodic_out = self.episodic_memory(
                current_feat,
                char_ids=char_ids,
                update_memory=update_episodic,
            )

        # Fuse outputs
        output, gate_weights = self.fusion(current_feat, working_out, episodic_out)

        return {
            "output": output,
            "working_out": working_out,
            "episodic_out": episodic_out,
            "gate_weights": gate_weights,
        }
