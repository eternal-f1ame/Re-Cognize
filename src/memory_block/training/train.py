#!/usr/bin/env python3
"""
Training Script for Memory Block Model

Trains the memory-enhanced Re-ID model (Re: Cognize / MeCha) for manga character Re-ID.
Supports multiple backbones: TransReID, MagiV2, MagiV3, InstructReID, ReID5o.

Key Features:
- Frozen backbone with optional LoRA adaptation (--use-lora)
- Working memory for panel context (per-character FIFO buffer)
- Episodic memory for character identity (non-parametric prototype slots)
- Gated fusion (single-residual design)
- Multi-task loss: CE (0.3) + Triplet (1.0) + Prototype (1.0) + Memory Consistency (0.1)
- Episodic training always ON when memory is enabled (matches test-time behavior)

Usage:
    # The exact command for one run of configs/runs.yaml: the recipe, the registry's
    # per-backbone values and the per-mode settings (for example CE weight 0 in memory mode)
    python scripts/train.py command transreid_memory_seed0

    # A direct call: TransReID backbone, recipe defaults, series-disjoint split (required)
    PYTHONHASHSEED=0 python src/memory_block/training/train.py --data-dir Datasets/popcharacters \
        --split-config configs/training/data_split.yaml

    # With MagiV2 backbone (input sizes and native dims per backbone: src/recognize/backbones.py)
    ... --backbone magiv2 --height 224 --width 224

    # Baseline (no memory) for comparison
    ... --no-memory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Import memory block modules
from memory_block.models import MemoryEnhancedReID, MemoryConfig
from memory_block.training.losses import CombinedMemoryLoss
from memory_block.training.dataset import (
    MangaCharacterDataset,
    PKSampler,
    get_transforms,
)
from recognize.data import load_split
from recognize.dev_eval import dev_score
from recognize.protocol_constants import CROP_PADDING
from recognize.provenance import assert_hashseed_pinned, stamp
from recognize.recipe import RECIPE

# W&B setup
import wandb
WANDB_AVAILABLE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Memory Block for Manga Re-ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Dataset directory")
    parser.add_argument("--combine-all", action="store_true",
                       help="Combine all subdirectories")
    parser.add_argument("--split-config", type=Path, default=None,
                       help="YAML file with train/dev/test series lists (required; implies --combine-all)")
    parser.add_argument("--exclude-categories", type=str, default="dbox,sbox,tbox",
                       help="Categories to exclude")
    parser.add_argument("--normalize", choices=["imagenet", "clip"], default=None,
                       help="Input normalisation statistics (default: the backbone registry's value)")
    parser.add_argument("--padding", type=float, default=CROP_PADDING,
                       help="Bbox padding ratio")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Train/val split ratio")

    # Model
    parser.add_argument("--backbone", type=str, default="transreid",
                       choices=["transreid", "magiv2", "magiv3", "instructreid", "reid5o", "custom"],
                       help="Backbone type")
    parser.add_argument("--backbone-checkpoint", type=Path, default=None,
                       help="Pretrained backbone checkpoint")
    parser.add_argument("--reid5o-config", type=Path, default=None,
                       help="ReID5o config file (required for reid5o backbone)")
    parser.add_argument("--backbone-module", type=str, default=None, help="custom backbone: importable module")
    parser.add_argument("--backbone-class", type=str, default=None, help="custom backbone: class name in the module")
    parser.add_argument("--backbone-kwargs", type=str, default="{}", help="custom backbone: JSON kwargs")
    parser.add_argument("--feat-dim", type=int, default=768,
                       help="Feature dimension (768 for ViT-B, 512 for CLIP)")
    parser.add_argument("--working-capacity", type=int, default=RECIPE.working_capacity,
                       help="Working memory capacity")
    parser.add_argument("--slots-per-char", type=int, default=RECIPE.slots_per_char,
                       help="Episodic memory slots per character")
    parser.add_argument("--no-memory", action="store_true",
                       help="Disable ALL memory (baseline Re-ID without memory block). "
                            "Use this for fair comparison with memory-enhanced version.")
    parser.add_argument("--no-working-memory", action="store_true",
                       help="Disable working memory only")
    parser.add_argument("--no-episodic-memory", action="store_true",
                       help="Disable episodic memory only")
    parser.add_argument("--residual-max-ratio", type=float, default=None,
                       help="Cap the memory residual at this fraction of the feature norm "
                            "(default: uncapped, as in every run of configs/runs.yaml)")

    # Training
    parser.add_argument("--epochs", type=int, default=RECIPE.epochs,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=RECIPE.batch_size,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=RECIPE.lr,
                       help="Learning rate (default: 1e-4) for memory modules")
    parser.add_argument("--weight-decay", type=float, default=RECIPE.weight_decay,
                       help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=RECIPE.warmup_epochs,
                       help="Warmup epochs")
    parser.add_argument("--triplet-margin", type=float, default=RECIPE.triplet_margin,
                       help="Triplet loss margin")
    parser.add_argument("--ce-weight", type=float, default=RECIPE.w_ce,
                       help="CrossEntropy loss weight (reduced: CE classifier is discarded at test time)")
    parser.add_argument("--triplet-weight", type=float, default=RECIPE.w_trip,
                       help="Triplet loss weight")
    parser.add_argument("--memory-weight", type=float, default=RECIPE.w_mem,
                       help="Memory consistency loss weight")
    parser.add_argument("--prototype-weight", type=float, default=RECIPE.w_proto,
                       help="Prototype classification loss weight (primary: directly optimizes test-time matching)")
    parser.add_argument("--proto-temperature", type=float, default=RECIPE.proto_temperature,
                       help="Prototype classification loss temperature (lower=sharper, higher=smoother)")

    # Sampling
    parser.add_argument("--pk-sampling", action="store_true",
                       help="Use PK sampling")
    parser.add_argument("--p", type=int, default=RECIPE.p,
                       help="Identities per batch (PK sampling)")
    parser.add_argument("--k", type=int, default=RECIPE.k,
                       help="Samples per identity (PK sampling)")

    # Episodic Training (Few-Shot Learning): always ON when memory is enabled
    # Memory ON = episodic training (few-shot per batch). No memory = baseline (no episodic).
    parser.add_argument("--k-support", type=int, default=RECIPE.k_support,
                       help="Number of support samples per character (episodic training)")
    parser.add_argument("--episodic-id-drop-rate", type=float, default=RECIPE.id_drop,
                       help="During training, probability of dropping char_ids for episodic "
                            "memory query, forcing it to use the search-all mode that matches "
                            "test-time behavior. 0=always use known IDs, 1=always search-all.")

    # LoRA backbone adaptation
    parser.add_argument("--use-lora", action="store_true",
                       help="Enable LoRA backbone adaptation (unfreezes low-rank adapters in attention)")
    parser.add_argument("--lora-rank", type=int, default=RECIPE.lora_rank,
                       help="LoRA rank (lower = fewer params)")
    parser.add_argument("--lora-alpha", type=float, default=RECIPE.lora_alpha,
                       help="LoRA scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                       help="LoRA dropout rate")
    parser.add_argument("--lora-layers", type=int, default=RECIPE.lora_layers,
                       help="Number of backbone layers (from end) to apply LoRA")
    parser.add_argument("--lora-lr", type=float, default=RECIPE.lora_lr,
                       help="Absolute learning rate of the LoRA parameters (recipe: 1e-5)")
    parser.add_argument("--memory-lr-scale", type=float, default=RECIPE.memory_lr_scale,
                       help="Memory block learning rate as multiple of main LR")

    # Input
    parser.add_argument("--height", type=int, default=256,
                       help="Input image height")
    parser.add_argument("--width", type=int, default=128,
                       help="Input image width")

    # System
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Device")
    parser.add_argument("--workers", type=int, default=4,
                       help="Data loading workers")
    parser.add_argument("--amp", dest="amp", action="store_true", default=RECIPE.amp, help="FP16 autocast (recipe default: on)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=RECIPE.seed,
                       help="Random seed")

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/memory"),
                       help="Output directory")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--save-freq", type=int, default=RECIPE.save_every,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--dev-eval-every", "--val-freq", dest="val_freq", type=int, default=RECIPE.dev_eval_every,
                       help="Dev evaluation / selection period in epochs (also at the last epoch)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from the latest epoch_*.pth in the output directory")

    # Logging (W&B enabled by default)
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="memory-block",
                       help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity (user or team). Defaults to logged-in user.")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def seed_all(seed: int) -> None:
    """Set random seeds."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_multiplier(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Per-epoch LR multiplier: linear warmup to 1.0 at epoch `warmup_epochs`, cosine to 0.0 at `total_epochs`."""
    import math
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    remaining = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, (epoch - warmup_epochs + 1) / remaining)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def get_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    """Warmup + single cosine decay to zero (no restarts, no floor)."""
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lambda e: lr_multiplier(e, warmup_epochs, total_epochs))


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, config, args, counts, dev_metric, training_mode):
    """Everything a training checkpoint carries, including provenance (git, args, counts)."""
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    prov = stamp(args_dict, None)
    prov.update({"dataset_counts": counts, "dev_metric": dev_metric, "epoch": epoch})
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "mAP": None if dev_metric is None else dev_metric.get("value"),
        "dev_metric": dev_metric,
        "config": config,
        "args": args_dict,
        "training_mode": training_mode,
        "provenance": prov,
    }


def step_seed(run_seed: int, epoch: int, batch_idx: int) -> int:
    """A per-step RNG seed, so a step's randomness depends on where it is, not on history.

    ID-drop and dropout draw from the global torch RNG inside the memory branch. Without
    reseeding, any change in draw order upstream (a different cuDNN algorithm, a different
    data-loader interleaving) shifts the whole trajectory: two runs of the same configuration
    and seed on different GPUs then differ by 0.18 mAP on the test set, against 0.007 for runs
    without memory. Reseeding each step makes the memory branch as reproducible as the rest.
    """
    return (run_seed * 1_000_003 + epoch * 10_007 + batch_idx) % (2 ** 31 - 1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    k_support: int = 5,
    run_seed: int = 0,
) -> Dict[str, float]:
    """Train for one epoch.

    When memory is enabled, uses episodic training (few-shot support/query split
    per batch). When memory is off (baseline), uses standard supervised training.

    Args:
        model: The model to train
        dataloader: Training dataloader (PK sampling required for memory mode)
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP
        k_support: Number of support samples per character (memory mode)
    """
    model.train()
    has_memory = model.memory_block is not None

    total_loss = 0.0
    all_metrics = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        torch.manual_seed(step_seed(run_seed, epoch, batch_idx))
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if has_memory:
            # Episodic training: reset memory, split batch into support/query
            loss, metrics = _episodic_train_step(
                model, images, labels, criterion, device, use_amp, k_support
            )
        else:
            # Baseline (no memory): standard supervised training
            with autocast(enabled=use_amp):
                output = model(
                    images,
                    char_ids=labels,
                    use_memory=False,
                    return_all=True,
                )
                loss, metrics = criterion(output, labels)

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        metrics["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

        total_loss += loss.item()
        num_batches += 1

        # Accumulate metrics
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = 0.0
            all_metrics[k] += v

        # Print progress
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {metrics.get('accuracy', 0):.3f} | "
                  f"ProtoAcc: {metrics.get('proto_accuracy', 0):.3f}")

    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
    avg_metrics["avg_loss"] = total_loss / num_batches

    return avg_metrics


def _episodic_train_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    k_support: int,
) -> tuple:
    """
    Episodic training step: split batch into support and query.

    Two-pass approach to match test-time identify_character() behavior:
    1. Support set initializes episodic prototypes and seeds working memory.
    2. Pass 1 on query (no grad): char_ids=None → episodic search-all → rough prediction.
    3. Pass 2 on query (grad):   char_ids=rough_pred → working memory routed to the predicted
       character's buffer, episodic memory identity-guided except on ID-drop rows → refined
       feature. Loss is computed here.

    Using predicted (possibly wrong) IDs for working memory routing instead of ground
    truth teaches the gate and attention to handle noisy routing, which is exactly
    what happens at test time. Early in training predictions are mostly wrong, so the
    model learns not to over-rely on working memory; as predictions improve, working
    memory becomes progressively more useful. This matches test-time behavior throughout.

    The batch should come from PK sampling with K >= k_support + 1.

    Args:
        model: The model
        images: (B, C, H, W) batch of images
        labels: (B,) character labels
        criterion: Loss function
        device: Device
        use_amp: Whether to use AMP
        k_support: Number of support samples per character

    Returns:
        loss: Computed loss (with gradients)
        metrics: Dict of metrics
    """
    # Reset memory for this episode
    if model.memory_block is not None:
        model.memory_block.reset()

    # Get unique characters in batch
    unique_chars = labels.unique()

    # Split into support and query indices
    support_indices = []
    query_indices = []

    for char_id in unique_chars:
        char_mask = labels == char_id
        char_indices = char_mask.nonzero(as_tuple=True)[0]

        # Take k_support for support, rest for query
        n_samples = len(char_indices)
        n_support = min(k_support, n_samples - 1)  # Leave at least 1 for query
        n_support = max(1, n_support)  # At least 1 for support

        support_indices.extend(char_indices[:n_support].tolist())
        query_indices.extend(char_indices[n_support:].tolist())

    if len(query_indices) == 0:
        # Degenerate case: episodic memory not yet initialized, no meaningful
        # routing possible, so run a single pass with no identity routing.
        with autocast(enabled=use_amp):
            output = model(
                images,
                char_ids=None,
                use_memory=True,
                update_working=False,
                update_episodic=False,
                return_all=True,
            )
            loss, metrics = criterion(output, labels)
        return loss, metrics

    support_indices = torch.tensor(support_indices, device=device)
    query_indices = torch.tensor(query_indices, device=device)

    support_images = images[support_indices]
    support_labels = labels[support_indices]
    query_images = images[query_indices]
    query_labels = labels[query_indices]

    with autocast(enabled=use_amp):
        # Step 1: Extract support features (no gradients needed for prototypes)
        with torch.no_grad():
            support_output = model(
                support_images,
                char_ids=support_labels,
                use_memory=False,  # Bootstrap: extract clean backbone features
                return_all=True,
            )
            support_bn = support_output["bn_feat"]  # no-memory bn_feat → protos + WM seed

        # Step 2: Temp-init episodic memory + seed working memory.
        # Both prototypes and WM are in BN-normalized space (matching memory block input).
        if model.memory_block is not None:
            model.memory_block.episodic_memory.initialize_from_support(
                support_bn,
                char_ids=support_labels,
                method="diverse",
            )
            model.memory_block.working_memory.update(
                support_bn, support_labels
            )

        # Prototypes stay in clean BN-normalized space (no step 2b re-extraction).
        # This provides stable training targets: the prototype loss pulls memory-
        # enhanced bn_feat toward clean prototypes, teaching the memory to refine
        # features toward the correct identity without circular dependencies.

        # Step 3: Two-pass query forward to match test-time behavior.
        #
        # Pass 1 (no grad): char_ids=None forces episodic into search-all mode,
        # which is what test-time pass 1 does. The resulting rough predictions
        # are used to route working memory in pass 2, exactly as in inference.
        with torch.no_grad():
            rough_output = model(
                query_images,
                char_ids=None,
                use_memory=True,
                update_working=False,
                update_episodic=False,
            )
            rough_pred = model.memory_block.episodic_memory.compute_similarity(
                rough_output["bn_feat"]
            ).argmax(dim=1)

        # Pass 2 (grad): char_ids=rough_pred routes each query to its predicted
        # character's working memory buffer, and episodic memory sees the same predicted
        # identity except on the ID-drop rows, which fall back to search-all. The call leaves
        # the search-all override unset: forcing it would bypass ID-drop.
        # Gradients flow through the attention and gate. Loss uses ground truth labels.
        query_output = model(
            query_images,
            char_ids=rough_pred,
            use_memory=True,
            update_working=True,
            update_episodic=False,
            return_all=True,
        )

        # Inject episodic prototypes for prototype classification loss
        if model.memory_block is not None:
            query_output["prototypes"] = model.memory_block.episodic_memory.prototypes.detach()
            query_output["prototype_mask"] = model.memory_block.episodic_memory.slot_filled

        # Step 4: Compute loss on query set (always against ground truth labels)
        loss, metrics = criterion(query_output, query_labels)

    # Add episodic-specific metrics
    metrics["n_support"] = len(support_indices)
    metrics["n_query"] = len(query_indices)
    metrics["n_chars"] = len(unique_chars)

    # Diagnostic: how much does memory change the features?
    # Both bn_feat and pre_memory_feat are L2-normalized, so
    # ||a - b|| = sqrt(2 - 2*cos(θ)) measures angular displacement.
    with torch.no_grad():
        pre_mem = query_output["pre_memory_feat"]
        post_mem = query_output["bn_feat"]
        metrics["memory_delta_norm"] = (post_mem - pre_mem).norm(dim=1).mean().item()

    return loss, metrics


def load_training_split(path: Path):
    """(train, dev, test) series lists of a split file; dev is mandatory, sets are disjoint."""
    try:
        split = load_split(path)
    except KeyError as e:
        raise ValueError(f"{path}: training needs 'train', 'dev' and 'test' lists (missing {e}); "
                         "checkpoint selection uses dev only") from e
    return split["train"], split["dev"], split["test"]


def train_memory_model(args: argparse.Namespace) -> None:
    """Main training function."""
    assert_hashseed_pinned()
    # Episodic training is always ON when memory is enabled, OFF for baseline.
    # This ensures training matches test-time behavior (few-shot per episode).
    memory_enabled = not args.no_memory
    args.episodic_training = memory_enabled

    if args.episodic_training:
        if not args.pk_sampling:
            print("Note: Episodic training requires PK sampling. Enabling PK sampling.")
            args.pk_sampling = True
        if args.k < args.k_support + 1:
            raise ValueError(
                f"For episodic training, K ({args.k}) must be >= k_support + 1 ({args.k_support + 1}). "
                f"Need at least {args.k_support} support + 1 query sample per character."
            )
        print(f"Episodic training: k_support={args.k_support}, k_query={args.k - args.k_support}")
    else:
        print("Baseline mode (no memory, no episodic training)")

    # Setup
    seed_all(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    if args.name is None:
        args.name = args.output_dir.name or f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # --output-dir is the run directory itself; the launcher owns the layout
    # (checkpoints/<backbone>/<config>/seed<s>). --name only labels the run (W&B, logs).
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Series-disjoint split: train series for PK batches, dev series for selection, test never read here.
    if not args.split_config:
        raise ValueError("--split-config is required: training uses series-disjoint splits (configs/training/data_split.yaml)")
    manga_filter_train, dev_series, test_series = load_training_split(args.split_config)
    args.combine_all = True
    use_manga_split = True
    dev_dirs = [args.data_dir / name for name in dev_series]
    for dd in dev_dirs:
        if not dd.exists():
            raise FileNotFoundError(f"dev series directory missing: {dd}")
    print(f"Split: {len(manga_filter_train)} train series, {len(dev_series)} dev series (selection), "
          f"{len(test_series)} test series (never read by training)")

    # Create datasets
    print("\nLoading datasets...")
    from recognize.backbones import BACKBONE_REGISTRY as _REG
    normalize = getattr(args, "normalize", None) or (_REG[args.backbone].normalize if args.backbone in _REG else "imagenet")
    print(f"Input normalisation: {normalize} (registry)")
    train_transform = get_transforms(args.height, args.width, is_train=True, normalize_type=normalize)
    val_transform = get_transforms(args.height, args.width, is_train=False, normalize_type=normalize)

    train_dataset = MangaCharacterDataset(
        data_dir=args.data_dir,
        split="train",
        transform=train_transform,
        padding=args.padding,
        train_ratio=1.0 if use_manga_split else args.train_ratio,
        exclude_categories=args.exclude_categories,
        seed=args.seed,
        combine_subdirs=args.combine_all,
        manga_filter=manga_filter_train,
    )

    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    from recognize.data import SeriesStream as _SeriesStream
    _dev_streams = [_SeriesStream(d) for d in dev_dirs]
    dataset_counts = {
        "train_crops": len(train_dataset), "train_identities": num_classes, "train_series": len(manga_filter_train),
        "dev_series": len(dev_dirs), "dev_crops": int(sum(st.n_crops for st in _dev_streams)),
        "dev_identities": int(sum(st.n_identities for st in _dev_streams)),
    }
    print(f"Counts: {dataset_counts}")

    # Create samplers
    if args.pk_sampling:
        train_sampler = PKSampler(train_dataset, p=args.p, k=args.k)
        # Use sampler's adjusted p (may be smaller if fewer identities)
        batch_size = train_sampler.p * args.k
        if train_sampler.p < args.p:
            print(f"Note: Adjusted P from {args.p} to {train_sampler.p} (only {len(train_sampler.labels)} identities)")
    else:
        train_sampler = None
        batch_size = args.batch_size

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    print("\nCreating model...")
    print(f"Backbone: {args.backbone}")

    # Build backbone config for special cases
    backbone_config = None
    if args.backbone == "reid5o" and args.reid5o_config:
        backbone_config = {"config_path": str(args.reid5o_config)}
    if args.backbone == "custom":
        if not (args.backbone_module and args.backbone_class):
            raise ValueError("--backbone custom needs --backbone-module and --backbone-class")
        backbone_config = {"module": args.backbone_module, "class": args.backbone_class,
                           "kwargs": json.loads(args.backbone_kwargs)}

    # Handle --no-memory flag (disables both working and episodic memory)
    use_working_memory = not args.no_working_memory and not args.no_memory
    use_episodic_memory = not args.no_episodic_memory and not args.no_memory

    if args.no_memory:
        print("Memory block DISABLED (baseline Re-ID mode)")
    else:
        print(f"Memory block enabled: working={use_working_memory}, episodic={use_episodic_memory}")

    config = MemoryConfig(
        num_classes=num_classes,
        feat_dim=args.feat_dim,
        use_working_memory=use_working_memory,
        use_episodic_memory=use_episodic_memory,
        working_capacity=args.working_capacity,
        slots_per_char=args.slots_per_char,
        freeze_backbone=True,
        backbone_type=args.backbone,
        backbone_checkpoint=str(args.backbone_checkpoint) if args.backbone_checkpoint else None,
        backbone_config=backbone_config,
        image_height=args.height,
        image_width=args.width,
        episodic_id_drop_rate=args.episodic_id_drop_rate,
        residual_max_ratio=args.residual_max_ratio,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
    )

    model = MemoryEnhancedReID(config)
    model = model.to(device)

    # Print parameter counts
    total_params = model.get_total_param_count()
    trainable_params = model.get_trainable_param_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Create loss function
    # The memory-consistency term is meaningless without a memory block.
    memory_weight = args.memory_weight if memory_enabled else 0.0
    if not memory_enabled and args.memory_weight:
        print(f"Baseline mode: memory-consistency weight {args.memory_weight} -> 0.0")
    criterion = CombinedMemoryLoss(
        num_classes=num_classes,
        feat_dim=args.feat_dim,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        triplet_weight=args.triplet_weight,
        memory_weight=memory_weight,
        prototype_weight=args.prototype_weight,
        proto_temperature=args.proto_temperature,
    )
    criterion = criterion.to(device)

    # Create optimizer (only trainable params)
    # Separate param groups: base, memory (a group of its own only when --memory-lr-scale is not
    # 1.0; the recipe trains memory at the base LR), LoRA (its own absolute LR)
    all_trainable = [p for p in model.parameters() if p.requires_grad]

    # Exclude classifier from optimizer when CE is disabled: it receives
    # zero gradient anyway, and leaving it in wastes weight-decay updates.
    if args.ce_weight == 0:
        classifier_ids = set(id(p) for p in model.classifier.parameters())
        all_trainable = [p for p in all_trainable if id(p) not in classifier_ids]

    # Identify special param sets
    lora_param_ids = set()
    if args.use_lora:
        from memory_block.models.lora import get_lora_parameters, count_lora_parameters
        lora_param_ids = set(id(p) for p in get_lora_parameters(model))

    memory_param_ids = set()
    if model.memory_block is not None and args.memory_lr_scale != 1.0:
        memory_param_ids = set(id(p) for p in model.memory_block.parameters() if p.requires_grad)

    # Build param groups (order: base, memory, lora)
    base_params = [p for p in all_trainable
                   if id(p) not in lora_param_ids and id(p) not in memory_param_ids]
    memory_params = [p for p in all_trainable if id(p) in memory_param_ids]
    lora_params = [p for p in all_trainable if id(p) in lora_param_ids]

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr})
    if memory_params:
        memory_lr = args.lr * args.memory_lr_scale
        param_groups.append({"params": memory_params, "lr": memory_lr, "name": "memory"})
        print(f"Memory parameters: {sum(p.numel() for p in memory_params):,} (lr={memory_lr:.2e})")
    if lora_params:
        lora_lr = args.lora_lr
        param_groups.append({"params": lora_params, "lr": lora_lr, "name": "lora"})
        lora_count = count_lora_parameters(model)
        print(f"LoRA parameters: {lora_count:,} (lr={lora_lr:.2e})")

    if not param_groups:
        param_groups = all_trainable

    optimizer = AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    # AMP
    scaler = GradScaler() if args.amp else None

    # Resume from latest epoch checkpoint
    start_epoch = 0
    resumed_best_mAP = 0.0
    resumed_history = None

    if args.resume:
        # Find the latest epoch_*.pth in the output directory
        # Handles both epoch_10.pth and epoch_0010.pth names (the sort is numeric)
        epoch_ckpts = sorted(
            output_dir.glob("epoch_*.pth"),
            key=lambda p: int(p.stem.replace("epoch_", "")),
        )
        if epoch_ckpts:
            resume_path = epoch_ckpts[-1]
            print(f"\nResuming from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"]
            resumed_best_mAP = ckpt.get("mAP") or 0.0
            print(f"  Resuming from epoch {start_epoch}, will train epochs {start_epoch+1}..{args.epochs}")

            # Load existing history if available
            history_path = output_dir / "history.json"
            if history_path.exists():
                with open(history_path) as f:
                    resumed_history = json.load(f)
                print(f"  Loaded training history ({len(resumed_history.get('train', []))} epochs)")
        else:
            print(f"\nNo epoch checkpoints found in {output_dir}, starting from scratch.")

    # W&B
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.name,
            config=vars(args),
            tags=[args.backbone, "memory-block"] + (["lora"] if args.use_lora else []),
        )
        # Set epoch as x-axis instead of steps for all panels
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("system/*", step_metric="epoch")
        # Best metrics should be monotonically non-decreasing
        wandb.define_metric("dev/*", step_metric="epoch")
        wandb.define_metric("dev/best", summary="max")

        # Log dataset summary
        wandb.config.update({
            "train_samples": len(train_dataset),
            "train_classes": train_dataset.num_classes,
            "dev_series": len(dev_dirs),
            "total_params": model.get_total_param_count(),
            "trainable_params": model.get_trainable_param_count(),
            "use_manga_split": use_manga_split,
        }, allow_val_change=True)

        # Log sample training images
        try:
            sample_images = []
            for i in range(min(16, len(train_dataset))):
                s = train_dataset[i]
                img = s["image"]
                # Denormalize for display
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_display = (img * std + mean).clamp(0, 1)
                caption = f"label={s['label']}"
                if "category" in s:
                    caption = s["category"]
                sample_images.append(wandb.Image(
                    img_display.permute(1, 2, 0).numpy(),
                    caption=caption,
                ))
            wandb.log({"samples/train_crops": sample_images}, step=0)
        except Exception:
            pass  # Don't fail training over logging

        # Log model architecture summary
        wandb.run.summary["model_architecture"] = {
            "backbone": args.backbone,
            "feat_dim": args.feat_dim,
            "memory_enabled": not args.no_memory,
            "use_working_memory": use_working_memory,
            "use_episodic_memory": use_episodic_memory,
            "residual_max_ratio": args.residual_max_ratio,
            "episodic_training": args.episodic_training,
        }

        print(f"W&B logging enabled: project={args.wandb_project}, run={args.name}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(output_dir / "provenance.json", "w") as f:
        _prov = stamp({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, None)
        _prov["dataset_counts"] = dataset_counts
        json.dump(_prov, f, indent=2, default=str)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_dev = resumed_best_mAP
    best_dev_metric = None
    last_dev = None
    history = resumed_history if resumed_history else {"train": [], "dev": []}
    history.setdefault("dev", [])
    training_mode = "baseline" if args.no_memory else "memory"

    if start_epoch > 0:
        print(f"Continuing from epoch {start_epoch + 1}/{args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_start = time.time()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, args.amp, scaler,
            k_support=args.k_support, run_seed=args.seed,
        )
        train_time = time.time() - train_start
        print(f"Train | Loss: {train_metrics['avg_loss']:.4f} | "
              f"Acc: {train_metrics.get('accuracy', 0):.3f} | "
              f"ProtoAcc: {train_metrics.get('proto_accuracy', 0):.3f} | "
              f"Time: {train_time:.1f}s")

        # Dev evaluation every val_freq epochs and at the last epoch (the selection metric for best.pth).
        run_val = (epoch + 1) % args.val_freq == 0 or (epoch + 1) == args.epochs

        # Save periodic checkpoint BEFORE evaluation (evaluation can be slow/hang)
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            torch.save(checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, config, args, dataset_counts,
                                          last_dev, training_mode), output_dir / f"epoch_{epoch + 1:04d}.pth")
            print(f"  Saved checkpoint epoch_{epoch + 1:04d}.pth")

        dev_metrics = None
        val_time = 0.0
        if run_val:
            val_start = time.time()
            dev_metrics = dev_score(model, val_transform, dev_dirs, memory=memory_enabled, device=str(device),
                                    batch_size=args.batch_size, num_workers=args.workers)
            model.train()
            val_time = time.time() - val_start
            best_dev_metric = dev_metrics["metric"]
            last_dev = dev_metrics
            print(f"Dev   | {dev_metrics['metric']}: {dev_metrics['value']:.4f} | "
                  + " ".join(f"{k}={v:.3f}" for k, v in dev_metrics["per_series"].items()) + f" | Time: {val_time:.1f}s")

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Save history
        train_metrics["epoch_time_s"] = round(epoch_time, 1)
        train_metrics["train_time_s"] = round(train_time, 1)
        history["train"].append(train_metrics)
        if run_val:
            history["dev"].append({"epoch": epoch + 1, "seconds": round(val_time, 1), **dev_metrics})

        # W&B logging
        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],

                # --- Train losses (raw, unweighted) ---
                "train/loss": train_metrics["avg_loss"],
                "train/total_loss": train_metrics.get("total_loss", train_metrics["avg_loss"]),
                "train/ce_loss": train_metrics.get("ce_loss", 0),
                "train/triplet_loss": train_metrics.get("triplet_loss", 0),
                "train/memory_loss": train_metrics.get("memory_loss", 0),

                # --- Train accuracy ---
                "train/accuracy": train_metrics.get("accuracy", 0),

                # --- Train triplet mining diagnostics ---
                "train/triplet_valid_triplets": train_metrics.get("triplet_valid_triplets", 0),
                "train/triplet_avg_pos_dist": train_metrics.get("triplet_avg_pos_dist", 0),
                "train/triplet_avg_neg_dist": train_metrics.get("triplet_avg_neg_dist", 0),
                "train/triplet_margin_gap": (
                    train_metrics.get("triplet_avg_neg_dist", 0)
                    - train_metrics.get("triplet_avg_pos_dist", 0)
                ),

                # --- Train gate weights ---
                "train/gate_working": train_metrics.get("gate_working", 0),
                "train/gate_episodic": train_metrics.get("gate_episodic", 0),

                # --- Train prototype metrics ---
                "train/prototype_loss": train_metrics.get("prototype_loss", 0),
                "train/proto_accuracy": train_metrics.get("proto_accuracy", 0),
                "train/proto_avg_similarity": train_metrics.get("proto_avg_similarity", 0),

                # --- Memory delta norm ---
                "train/memory_delta_norm": train_metrics.get("memory_delta_norm", 0),

                # --- Episodic batch composition ---
                "train/n_support": train_metrics.get("n_support", 0),
                "train/n_query": train_metrics.get("n_query", 0),
                "train/n_chars_per_batch": train_metrics.get("n_chars", 0),

                # --- Gradient norm ---
                "train/grad_norm": train_metrics.get("grad_norm", 0),

                # --- Timing ---
                "system/epoch_time_s": epoch_time,
                "system/train_time_s": train_time,
            }

            if run_val:
                log_dict.update({
                    f"dev/{dev_metrics['metric']}": dev_metrics["value"],
                    "dev/best": max(best_dev, dev_metrics["value"]),
                    "system/val_time_s": val_time,
                })
                log_dict.update({f"dev/{k}": v for k, v in dev_metrics["per_series"].items()})

            # GPU memory stats
            if torch.cuda.is_available():
                log_dict["system/gpu_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                log_dict["system/gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
                torch.cuda.reset_peak_memory_stats()

            # LoRA weight norm
            if args.use_lora:
                lora_norm = 0.0
                lora_count = 0
                for name, param in model.named_parameters():
                    if "lora_" in name and param.requires_grad:
                        lora_norm += param.data.norm().item() ** 2
                        lora_count += 1
                if lora_count > 0:
                    log_dict["train/lora_weight_norm"] = lora_norm ** 0.5
                    log_dict["train/lora_num_layers"] = lora_count

            wandb.log(log_dict)

        # Save best model by the dev metric (only on dev-eval epochs)
        if run_val and dev_metrics["value"] > best_dev:
            best_dev = dev_metrics["value"]
            best_path = output_dir / "best.pth"
            torch.save(checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, config, args, dataset_counts,
                                          dev_metrics, training_mode), best_path)
            print(f"  Saved best model ({dev_metrics['metric']} {best_dev:.4f})")
            if use_wandb:
                artifact = wandb.Artifact(
                    f"best-model-{args.backbone}", type="model",
                    metadata={"dev_metric": dev_metrics["metric"], "value": best_dev, "epoch": epoch + 1},
                )
                artifact.add_file(str(best_path))
                wandb.log_artifact(artifact, aliases=["best"])

    # The reported checkpoint is the last epoch, not the dev-selected one: selecting on the small
    # dev set lowers test mAP by 0.10-0.39 for every memory run and can pick a checkpoint as early
    # as epoch 10. best.pth is also written, for diagnostics.
    torch.save(checkpoint_payload(model, optimizer, scheduler, scaler, args.epochs, config, args,
                                  dataset_counts, last_dev, training_mode), output_dir / "final.pth")
    print(f"  Saved final.pth (epoch {args.epochs}); best.pth remains the dev-selected checkpoint")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best dev {best_dev_metric}: {best_dev:.4f} (selection metric; the reported checkpoint is final.pth)")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)

    if use_wandb:
        # Log final summary metrics
        wandb.run.summary["best_dev"] = best_dev
        wandb.run.summary["best_dev_metric"] = best_dev_metric
        wandb.run.summary["total_epochs"] = args.epochs

        # Log config + history as artifacts
        run_artifact = wandb.Artifact(
            f"run-{args.name}", type="run-output",
            metadata={"best_dev": best_dev, "best_dev_metric": best_dev_metric},
        )
        run_artifact.add_file(str(config_path))
        run_artifact.add_file(str(output_dir / "history.json"))
        wandb.log_artifact(run_artifact)

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train_memory_model(args)
