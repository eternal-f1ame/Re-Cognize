"""
Loss Functions for Memory Block Training

Includes:
- MemoryConsistencyLoss: Ensures retrieved memory matches ground truth identity
- CombinedMemoryLoss: Complete loss function for memory model training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss with cosine distance.

    Uses batch-hard mining to find the hardest positive and negative for each sample.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, D) embeddings (will be L2 normalized)
            labels: (B,) person IDs

        Returns:
            loss: Triplet loss value
            info: Diagnostic info dict
        """
        # L2 normalize for cosine distance
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Cosine distance
        dist = 1.0 - torch.mm(embeddings, embeddings.t()).clamp(-1, 1)
        N = dist.size(0)

        # Masks
        mask_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        mask_neg = ~mask_pos
        mask_pos.fill_diagonal_(False)

        # Valid samples
        has_pos = mask_pos.any(dim=1)
        has_neg = mask_neg.any(dim=1)
        valid = has_pos & has_neg

        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True), {
                "valid_triplets": 0
            }

        # Hardest positive
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = -float("inf")
        hardest_pos = dist_pos.max(dim=1)[0][valid]

        # Hardest negative
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = float("inf")
        hardest_neg = dist_neg.min(dim=1)[0][valid]

        # Clamp for stability
        hardest_pos = hardest_pos.clamp(0, 2)
        hardest_neg = hardest_neg.clamp(0, 2)

        # Loss
        target = torch.ones_like(hardest_pos)
        loss = self.ranking_loss(hardest_neg, hardest_pos, target)

        info = {
            "valid_triplets": valid.sum().item(),
            "avg_pos_dist": hardest_pos.mean().item(),
            "avg_neg_dist": hardest_neg.mean().item(),
        }

        return loss, info



class MemoryConsistencyLoss(nn.Module):
    """InfoNCE between memory-enhanced features and the pre-memory features of the same identity.

    The self pair (memory_i vs original_i) counts as a positive: the memory output must stay
    close to its own clean feature, and a batch with one crop per identity still produces a
    gradient. With the diagonal masked out of the positives (`include_self=False`), such a
    batch has no positive and the loss is exactly 0.

    Args:
        temperature: InfoNCE temperature.
        include_self: keep the self pair among the positives (default).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        include_self: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.include_self = include_self

    def forward(
        self,
        memory_output: torch.Tensor,
        original_feat: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory_output: (B, D) output from memory module
            original_feat: (B, D) original feature before memory
            labels: (B,) character IDs

        Returns:
            loss: Memory consistency loss
        """
        B = memory_output.shape[0]

        # Normalize features
        memory_norm = F.normalize(memory_output, p=2, dim=1)
        original_norm = F.normalize(original_feat, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(memory_norm, original_norm.t()) / self.temperature

        # Positives: same identity, self pair included unless disabled
        label_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
        if not self.include_self:
            label_mask.fill_diagonal_(0)

        # If no positives at all, there is nothing to pull together
        if label_mask.sum() == 0:
            return torch.tensor(0.0, device=memory_output.device, requires_grad=True)

        # InfoNCE (log-sum-exp for numerical stability). The denominator spans the same
        # candidate set as the positives, so the self pair is in both or in neither.
        masked_sim = sim_matrix.clone()
        if not self.include_self:
            masked_sim.fill_diagonal_(float("-inf"))
        log_denom = torch.logsumexp(masked_sim, dim=1)

        # Log of positive sum: log(sum_j exp(s_ij) * mask_ij)
        # Mask out non-positives by setting to -inf
        pos_masked_sim = sim_matrix.clone()
        pos_masked_sim[label_mask == 0] = float("-inf")
        log_pos = torch.logsumexp(pos_masked_sim, dim=1)

        # Loss = -mean(log_pos - log_denom)
        # Mask valid samples so isolated items don't cause NaN when their row is all -inf
        valid_samples = label_mask.sum(dim=1) > 0
        loss = -(log_pos[valid_samples] - log_denom[valid_samples]).mean()

        return loss


class PrototypeClassificationLoss(nn.Module):
    """
    Prototype-based Classification Loss.

    Aligns training with test-time behavior by computing cross-entropy
    over cosine similarities to episodic memory prototypes. At test time,
    characters are identified by cosine NN matching against prototypes, and
    this loss directly optimizes for that objective.

    For each query, computes max cosine similarity to each character's
    prototype slots, then applies temperature-scaled cross-entropy.

    Args:
        temperature: Temperature for scaling similarities (default: 0.15)
    """

    def __init__(self, temperature: float = 0.15) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_features: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
        prototype_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            query_features: (B, D) post-memory features
            labels: (B,) ground truth character IDs
            prototypes: (num_chars, slots, D) episodic memory prototypes
            prototype_mask: (num_chars, slots) bool, True where slot is filled

        Returns:
            loss: Prototype classification loss
            info: Dict with proto_accuracy and proto_avg_similarity
        """
        B, D = query_features.shape
        num_chars, slots_per_char, _ = prototypes.shape

        # Normalize
        query_norm = F.normalize(query_features, p=2, dim=-1)  # (B, D)
        proto_norm = F.normalize(
            prototypes.view(-1, D), p=2, dim=-1
        )  # (num_chars * slots, D)

        # Compute all similarities: (B, num_chars * slots)
        all_sims = torch.mm(query_norm, proto_norm.t())
        all_sims = all_sims.view(B, num_chars, slots_per_char)

        # Mask unfilled slots
        mask_exp = prototype_mask.unsqueeze(0).expand(B, -1, -1)  # (B, num_chars, slots)
        all_sims = all_sims.masked_fill(~mask_exp, float('-inf'))

        # Max similarity per character (most similar prototype)
        char_sims, _ = all_sims.max(dim=-1)  # (B, num_chars)

        # Mask characters with no prototypes at all
        has_protos = prototype_mask.any(dim=1)  # (num_chars,)
        char_sims = char_sims.masked_fill(
            ~has_protos.unsqueeze(0).expand(B, -1), float('-inf')
        )

        # Check that target labels have valid prototypes
        valid_targets = has_protos[labels]
        if not valid_targets.any():
            return torch.tensor(0.0, device=query_features.device, requires_grad=True), {
                "proto_accuracy": 0.0,
                "proto_avg_similarity": 0.0,
            }

        # Temperature-scaled cross-entropy
        logits = char_sims / self.temperature
        loss = F.cross_entropy(logits[valid_targets], labels[valid_targets])

        # Metrics, evaluated only over samples whose true class has prototypes.
        # For invalid samples char_sims is all-inf → argmax is undefined/arbitrary,
        # so including them would corrupt the diagnostic metric.
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions[valid_targets] == labels[valid_targets]).float().mean().item()
        avg_sim = char_sims[torch.arange(B, device=labels.device), labels]
        avg_sim = avg_sim[avg_sim > float('-inf')].mean().item() if valid_targets.any() else 0.0

        return loss, {
            "proto_accuracy": accuracy,
            "proto_avg_similarity": avg_sim,
        }



class CombinedMemoryLoss(nn.Module):
    """
    Combined Loss Function for Memory Model Training.

    Combines:
    - CrossEntropy loss (classification)
    - Triplet loss (metric learning)
    - Memory consistency loss (memory correctness)
    - Prototype classification loss (aligns training with test-time cosine matching)

    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension
        triplet_margin: Triplet loss margin
        label_smoothing: Label smoothing factor
        ce_weight: CrossEntropy loss weight
        triplet_weight: Triplet loss weight
        memory_weight: Memory consistency loss weight
        prototype_weight: Prototype classification loss weight
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 768,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.1,
        ce_weight: float = 0.3,
        triplet_weight: float = 1.0,
        memory_weight: float = 0.1,
        prototype_weight: float = 1.0,
        proto_temperature: float = 0.15,
        memory_consistency_self_pair: bool = True,
    ) -> None:
        super().__init__()

        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.memory_weight = memory_weight
        self.prototype_weight = prototype_weight

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet_loss = BatchHardTripletLoss(margin=triplet_margin)
        self.memory_loss = MemoryConsistencyLoss(include_self=memory_consistency_self_pair)
        self.prototype_loss = PrototypeClassificationLoss(temperature=proto_temperature)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            model_output: Dict from model forward pass containing:
                - logits: (B, C) classification logits
                - bn_feat: (B, D) BNNeck features (post-memory, re-normalized)
                - final_feat: (B, D) final features after memory
                - pre_memory_feat: (B, D) BNNeck features before memory
                - gate_weights: (B, 2) optional gate weights
            labels: (B,) ground truth labels

        Returns:
            total_loss: Combined loss
            loss_dict: Dict with individual loss values
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)

        # CrossEntropy loss (skipped when ce_weight=0, i.e. memory mode)
        logits = model_output["logits"]
        if self.ce_weight > 0:
            ce = self.ce_loss(logits, labels)
            total_loss = total_loss + self.ce_weight * ce
            loss_dict["ce_loss"] = ce.item()

        # Accuracy (always logged for diagnostics)
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean().item()
        loss_dict["accuracy"] = acc

        # Triplet loss on bn_feat (memory-enhanced, post-BNNeck)
        bn_feat = model_output["bn_feat"]
        triplet, triplet_info = self.triplet_loss(bn_feat, labels)
        total_loss = total_loss + self.triplet_weight * triplet
        loss_dict["triplet_loss"] = triplet.item()
        loss_dict.update({f"triplet_{k}": v for k, v in triplet_info.items()})

        # Memory consistency loss (final_feat vs pre_memory_feat, both in BN space).
        # Only for models that actually have a memory block: without one the two features are
        # the same tensor and the term would be a SupCon objective on the baseline.
        # `gate_weights` is present exactly when the memory block ran.
        pre_memory = model_output.get("pre_memory_feat")
        has_memory_path = model_output.get("gate_weights") is not None
        if (self.memory_weight > 0 and has_memory_path and pre_memory is not None
                and model_output.get("final_feat") is not None):
            memory = self.memory_loss(
                model_output["final_feat"],
                pre_memory,
                labels,
            )
            total_loss = total_loss + self.memory_weight * memory
            loss_dict["memory_loss"] = memory.item()

        # Gate weight logging (no regularization; the gate stays adaptive)
        gate_weights = model_output.get("gate_weights")
        if gate_weights is not None:
            loss_dict["gate_working"] = gate_weights[:, 0].mean().item()
            loss_dict["gate_episodic"] = gate_weights[:, 1].mean().item()

        # Prototype classification loss (aligns training with test-time cosine matching).
        # Uses bn_feat (backbone+memory+BNNeck) as query, matching the bn_feat space used
        # by compute_similarity() in evaluation, so prototypes and queries are in the same
        # BNNeck-normalized space at both train and test time.
        prototypes = model_output.get("prototypes")
        prototype_mask = model_output.get("prototype_mask")
        if (
            self.prototype_weight > 0
            and prototypes is not None
            and prototype_mask is not None
            and "bn_feat" in model_output
        ):
            proto_loss, proto_info = self.prototype_loss(
                model_output["bn_feat"],
                labels,
                prototypes,
                prototype_mask,
            )
            total_loss = total_loss + self.prototype_weight * proto_loss
            loss_dict["prototype_loss"] = proto_loss.item()
            loss_dict.update(proto_info)

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
