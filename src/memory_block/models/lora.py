"""
LoRA (Low-Rank Adaptation) for backbone domain adaptation.

Injects small trainable rank-decomposition matrices into frozen backbone
attention layers, enabling manga/comic domain adaptation with minimal
parameter overhead (~0.5% of backbone params).

Usage:
    from memory_block.models.lora import inject_lora, get_lora_parameters

    # After freezing backbone
    injected = inject_lora(backbone, backbone_type="transreid", rank=8)
    lora_params = get_lora_parameters(backbone)
    optimizer.add_param_group({"params": lora_params, "lr": 1e-5})   # the recipe's lora_lr
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear layer.

    Computes: y = W_frozen @ x + bias + (x @ A^T @ B^T) * scaling
    where A is (rank, in_features) and B is (out_features, rank).

    At initialization B=0, so the LoRA contribution is zero and the
    model produces identical outputs to the original frozen linear.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Store original linear (frozen: its parameters stay requires_grad=False)
        self.original = original_linear

        # LoRA matrices (trainable), always float32 for stable gradients
        # (GradScaler requires fp32 grads).  We cast to input dtype in forward().
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Dropout on input before LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize A with kaiming, B with zeros → initial LoRA = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B already zeros

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight = W_frozen + B @ A * scaling.

        Exposed so that nn.MultiheadAttention (which accesses out_proj.weight
        directly in its forward) can read the combined weight matrix.
        Cast LoRA delta to frozen weight dtype (handles fp16 backbones).
        """
        w = self.original.weight
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return w + delta.to(w.dtype)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Pass through the original bias (LoRA doesn't add a bias term)."""
        return self.original.bias

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Frozen layer plus the low-rank term.

        Extra arguments are handed to the wrapped layer untouched. ReID5o's own
        `LoRALinear` is an `nn.Linear` subclass whose forward takes a `lora_index`
        selecting a released modality adapter; wrapping it transparently keeps those
        pretrained weights in the path and adds our adapter on top. Dropping the
        argument (or refusing it) would silently discard them.
        """
        result = self.original(x, *args, **kwargs)
        # LoRA path, cast to input dtype for fp16 backbones
        lora_out = self.lora_dropout(x) @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T
        return result + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.2f}"
        )


def _get_target_modules_transreid(
    model: nn.Module,
    num_layers: int,
) -> List[str]:
    """Get target module names for TransReID/InstructReID (timm ViT)."""
    # Find all blocks
    block_names = []
    for name, mod in model.named_modules():
        if re.match(r".*blocks\.\d+$", name):
            block_names.append(name)

    if not block_names:
        # Try backbone.blocks pattern
        for name, mod in model.named_modules():
            if re.match(r".*backbone\.blocks\.\d+$", name):
                block_names.append(name)

    if not block_names:
        return []

    # Sort by block index
    block_names.sort(key=lambda n: int(n.split(".")[-1]))

    # Take last N blocks
    target_blocks = block_names[-num_layers:]

    targets = []
    for block_name in target_blocks:
        targets.append(f"{block_name}.attn.qkv")
        targets.append(f"{block_name}.attn.proj")
    return targets


def _get_target_modules_magiv2(
    model: nn.Module,
    num_layers: int,
) -> List[str]:
    """Get target module names for MagiV2 (HuggingFace ViT)."""
    layer_names = []
    for name, mod in model.named_modules():
        if re.match(r".*encoder\.layer\.\d+$", name):
            layer_names.append(name)

    if not layer_names:
        return []

    layer_names.sort(key=lambda n: int(n.split(".")[-1]))
    target_layers = layer_names[-num_layers:]

    targets = []
    for layer_name in target_layers:
        targets.append(f"{layer_name}.attention.attention.query")
        targets.append(f"{layer_name}.attention.attention.key")
        targets.append(f"{layer_name}.attention.attention.value")
        targets.append(f"{layer_name}.attention.output.dense")
    return targets


def _get_target_modules_magiv3(
    model: nn.Module,
    num_layers: int,
) -> List[str]:
    """Get target module names for MagiV3 (Florence2 vision encoder)."""
    # Florence2 vision encoder uses a DaViT-like structure
    layer_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "attn" in name.lower():
            # Collect parent layer indices
            layer_names.append(name)

    if not layer_names:
        # Fallback: find any attention linear layers
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name):
                layer_names.append(name)

    # Take last N targets
    if len(layer_names) > num_layers * 4:
        layer_names = layer_names[-(num_layers * 4):]

    return layer_names


def _get_target_modules_reid5o(
    model: nn.Module,
    num_layers: int,
) -> List[str]:
    """Get target module names for ReID5o (CLIP visual transformer)."""
    # CLIP uses resblocks with fused in_proj_weight (not nn.Linear for QKV)
    # Target the out_proj and MLP layers instead
    block_names = []
    for name, mod in model.named_modules():
        if re.match(r".*resblocks\.\d+$", name):
            block_names.append(name)

    if not block_names:
        # Try visual.transformer.resblocks
        for name, mod in model.named_modules():
            if re.match(r".*transformer\.resblocks\.\d+$", name):
                block_names.append(name)

    if not block_names:
        return []

    block_names.sort(key=lambda n: int(n.split(".")[-1]))
    target_blocks = block_names[-num_layers:]

    targets = []
    for block_name in target_blocks:
        # CLIP uses nn.MultiheadAttention with in_proj_weight (not nn.Linear)
        # Target out_proj and MLP fc layers
        for name, mod in model.named_modules():
            if name.startswith(block_name) and isinstance(mod, nn.Linear):
                targets.append(name)

    return targets


def _block_prefix(name: str) -> Optional[str]:
    """The enclosing block of a module name: everything up to and including its last integer segment.

    'blocks.11.attn.qkv' -> 'blocks.11'; 'vision_tower.blocks.3.0.channel_block.channel_attn.fn.qkv'
    -> 'vision_tower.blocks.3.0', so nested stages are handled without knowing the topology.
    """
    parts = name.split(".")
    idx = max((i for i, part in enumerate(parts) if part.isdigit()), default=None)
    return None if idx is None else ".".join(parts[: idx + 1])


def used_linear_modules(model: nn.Module, probe_forward) -> List[str]:
    """Names of the nn.Linear modules that `probe_forward()` actually calls.

    Multi-modal backbones carry linears the image path never touches (Florence-2's
    language decoder, ReID5o's text and cross-modal encoders). Adapters placed there
    receive no gradient and change no feature: the LoRA row would differ from its
    non-LoRA row by noise alone.
    """
    fired: List[str] = []
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: fired.append(n)))
    try:
        with torch.no_grad():
            probe_forward()
    finally:
        for h in hooks:
            h.remove()
    seen, out = set(), []
    for n in fired:                                   # keep call order, drop repeats
        if n not in seen:
            seen.add(n); out.append(n)
    return out


# Attention projections, the recipe's LoRA target ("r=8, alpha=16, last L=4 layers").
ATTENTION_HINTS = ("attn", "attention", "qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj_q", "proj_k", "proj_v")


def _targets_from_probe(used: List[str], num_layers: int, attention_only: bool = True) -> List[str]:
    """Attention projections of the last `num_layers` blocks the probe actually executed.

    `used` is in call order, so "the last N blocks" is the last N distinct blocks to run: the
    deepest ones, whatever the topology (a flat ViT stack, DaViT's nested stages, CLIP resblocks).
    Restricting to attention projections keeps the recipe's meaning of "LoRA on the last L layers";
    a trunk whose attention is not an nn.Linear (ReID5o fuses its in_proj) keeps its block linears.
    """
    order: List[str] = []
    for name in used:
        prefix = _block_prefix(name)
        if prefix is not None and (not order or order[-1] != prefix) and prefix not in order:
            order.append(prefix)
    chosen = set(order[-num_layers:])
    if not chosen:
        return []
    names = [n for n in used if _block_prefix(n) in chosen]
    if not attention_only:
        return names
    attn = [n for n in names if any(h in n.lower() for h in ATTENTION_HINTS)]
    return attn or names


def inject_lora(
    model: nn.Module,
    backbone_type: str = "transreid",
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    num_layers: int = 4,
    probe_forward=None,
) -> List[str]:
    """Inject LoRA adapters into backbone attention layers.

    Args:
        model: The backbone model (not the full MemoryEnhancedReID)
        backbone_type: One of transreid, magiv2, magiv3, instructreid, reid5o
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        dropout: Dropout on LoRA input path
        num_layers: Number of backbone layers (from end) to apply LoRA to

    Returns:
        List of module names that were injected with LoRA
    """
    # Get target modules based on backbone type
    used = used_linear_modules(model, probe_forward) if probe_forward is not None else None
    if used is not None:
        probe_targets = _targets_from_probe(used, num_layers)
        if probe_targets:
            targets = probe_targets
            injected = _replace(model, targets, rank, alpha, dropout)
            print(f"LoRA injected into {len(injected)} layers of the image path "
                  f"(rank={rank}, alpha={alpha}; {len(used)} linears ran in the probe)")
            return injected
        print("Warning: the LoRA probe found no transformer blocks; falling back to the per-backbone rule")

    if backbone_type in ("transreid", "instructreid"):
        targets = _get_target_modules_transreid(model, num_layers)
    elif backbone_type == "magiv2":
        targets = _get_target_modules_magiv2(model, num_layers)
    elif backbone_type == "magiv3":
        targets = _get_target_modules_magiv3(model, num_layers)
    elif backbone_type == "reid5o":
        targets = _get_target_modules_reid5o(model, num_layers)
    else:
        print(f"Warning: Unknown backbone type '{backbone_type}' for LoRA injection")
        return []

    if not targets:
        print(f"Warning: No target modules found for LoRA injection in {backbone_type}")
        return []

    if used is not None:
        dropped = [t for t in targets if t not in set(used)]
        targets = [t for t in targets if t in set(used)]
        if dropped:
            print(f"LoRA: {len(dropped)} target(s) skipped, the image path never calls them (e.g. {dropped[0]})")
    return _replace(model, targets, rank, alpha, dropout)


def _replace(model: nn.Module, targets: List[str], rank: int, alpha: float, dropout: float) -> List[str]:
    """Wrap each named nn.Linear in `targets` with a LoRALinear; return the names wrapped."""
    def _get(container, key):
        """Child by name, whether the parent is a Module, a Sequential/ModuleList or a ModuleDict."""
        if hasattr(container, key):
            return getattr(container, key)
        return container[int(key)] if key.isdigit() and not isinstance(container, nn.ModuleDict) else container[key]

    def _set(container, key, value):
        if isinstance(container, nn.ModuleDict):
            container[key] = value
        elif key.isdigit() and not hasattr(container, key):
            container[int(key)] = value
        else:
            setattr(container, key, value)

    injected = []
    for target_name in targets:
        parts = target_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = _get(parent, part)
        attr_name = parts[-1]
        original = _get(parent, attr_name)

        if not isinstance(original, nn.Linear) or isinstance(original, LoRALinear):
            continue

        _set(parent, attr_name, LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout))
        injected.append(target_name)

    return injected


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect all LoRA parameters from a model.

    Returns:
        List of trainable LoRA parameters (lora_A and lora_B)
    """
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def count_lora_parameters(model: nn.Module) -> int:
    """Count total number of LoRA parameters."""
    return sum(p.numel() for p in get_lora_parameters(model))


def merge_lora(model: nn.Module) -> None:
    """Merge LoRA weights into original linear layers for inference speed.

    After merging, LoRA modules are replaced with plain nn.Linear.
    This is irreversible: save the checkpoint before merging if needed.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            # Compute merged weight: W_new = W + B @ A * scaling
            with torch.no_grad():
                merged_weight = (
                    module.original.weight
                    + (module.lora_B @ module.lora_A) * module.scaling
                )
                module.original.weight.copy_(merged_weight)

            # Replace LoRA module with the original (now merged) linear
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module.original)


if __name__ == "__main__":
    """Quick self-test."""
    print("=" * 60)
    print("LoRA Module Self-Test")
    print("=" * 60)

    # Test 1: LoRALinear produces identical output at init (B=0)
    print("\nTest 1: Zero-init check...")
    linear = nn.Linear(768, 768)
    linear.weight.requires_grad = False
    if linear.bias is not None:
        linear.bias.requires_grad = False

    lora = LoRALinear(linear, rank=8, alpha=16.0)
    x = torch.randn(4, 768)

    with torch.no_grad():
        out_original = linear(x)
        out_lora = lora(x)

    diff = (out_original - out_lora).abs().max().item()
    assert diff < 1e-6, f"Zero-init failed: max diff = {diff}"
    print(f"  [OK] Max diff = {diff:.2e}")

    # Test 2: LoRA params are trainable, original is frozen
    print("\nTest 2: Gradient check...")
    x = torch.randn(4, 768, requires_grad=True)
    out = lora(x)
    out.sum().backward()
    assert lora.lora_A.grad is not None, "lora_A has no gradient"
    assert lora.lora_B.grad is not None, "lora_B has no gradient"
    assert lora.original.weight.grad is None, "Original weight should have no gradient"
    print("  [OK] LoRA params have gradients, original frozen")

    # Test 3: Inject into a simple ViT-like model
    print("\nTest 3: Injection into timm ViT...")
    from timm.models.vision_transformer import vit_base_patch16_224
    vit = vit_base_patch16_224(pretrained=False)
    total_before = sum(p.numel() for p in vit.parameters())

    # Freeze all params first (simulating freeze_backbone=True)
    for p in vit.parameters():
        p.requires_grad = False

    injected = inject_lora(vit, backbone_type="transreid", rank=8, num_layers=4)
    print(f"  Injected into {len(injected)} modules:")
    for name in injected:
        print(f"    - {name}")

    lora_params = get_lora_parameters(vit)
    lora_count = sum(p.numel() for p in lora_params)
    print(f"  LoRA params: {lora_count:,} ({100 * lora_count / total_before:.2f}% of backbone)")

    # Verify only LoRA params are trainable
    trainable = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    assert trainable == lora_count, f"Trainable ({trainable}) != LoRA ({lora_count})"
    print(f"  [OK] Only LoRA params are trainable")

    # Test 4: Forward pass works
    print("\nTest 4: Forward pass...")
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = vit(dummy)
    print(f"  Output shape: {out.shape}")
    print("  [OK] Forward pass successful")

    # Test 5: merge_lora
    print("\nTest 5: Merge LoRA...")
    # Make LoRA do something non-zero first
    for p in lora_params:
        p.data.fill_(0.01)

    with torch.no_grad():
        out_before = vit(dummy)

    merge_lora(vit)
    lora_after = get_lora_parameters(vit)
    assert len(lora_after) == 0, "LoRA modules still present after merge"

    with torch.no_grad():
        out_after = vit(dummy)

    merge_diff = (out_before - out_after).abs().max().item()
    print(f"  Max diff after merge: {merge_diff:.2e}")
    assert merge_diff < 1e-4, f"Merge changed output too much: {merge_diff}"
    print("  [OK] Merge successful, outputs match")

    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print(f"{'=' * 60}")
