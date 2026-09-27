"""
Memory-Enhanced Re-ID Model for Manga Character Re-ID

This is the main integrated model that combines:
- Any backbone (TransReID, MagiV2, MagiV3, InstructReID, ReID5o)
- Working memory (panel context)
- Episodic memory (character identity)
- Gated fusion (adaptive combination)

The backbone is frozen. Training updates the BNNeck, the classifier (when its loss
weight is non-zero), the memory block when present, and LoRA adapters when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal, Union, Any
import sys

import torch
from recognize.loading import load_state_dict_checked
from recognize.backbones import BACKBONE_REGISTRY
from .key_maps import infer_pos_grid, map_instructreid_keys, map_transreid_keys
from .mae_masking import disable_on_backbone, mask_ratio_of
import torch.nn as nn
import torch.nn.functional as F

from .memory_modules import MemoryBlock


@dataclass
class MemoryConfig:
    """Configuration for Memory-Enhanced Re-ID model."""

    # Model dimensions
    feat_dim: int = 768
    num_classes: int = 100  # Will be set based on dataset

    # Working memory
    working_capacity: int = 8
    use_working_memory: bool = True

    # Episodic memory
    slots_per_char: int = 5
    use_episodic_memory: bool = True
    episodic_id_drop_rate: float = 0.5  # Drop char_ids for episodic query this fraction of training steps

    # Attention
    num_heads: int = 8
    dropout: float = 0.1

    # Fusion: cap on how far memory may move a feature, as a fraction of that feature's norm.
    # None is the unconstrained residual: every run in configs/runs.yaml trains with it, and a
    # saved config without this field loads with it.
    residual_max_ratio: Optional[float] = None

    # Backbone configuration
    backbone_type: Literal["transreid", "magiv2", "magiv3", "instructreid", "reid5o", "custom"] = "transreid"
    freeze_backbone: bool = True
    backbone_checkpoint: Optional[str] = None
    backbone_config: Optional[Dict[str, Any]] = None  # Additional backbone config

    # LoRA backbone adaptation
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_layers: int = 4  # Number of backbone layers (from end) to apply LoRA

    # Input
    image_height: int = 256
    image_width: int = 128


class BNNeck(nn.Module):
    """BatchNorm neck as in TransReID."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim, affine=True)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.bias, 0.0)
        nn.init.constant_(self.bn.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.bn(x), p=2, dim=1)


class BackboneWrapper(nn.Module):
    """
    Universal backbone wrapper that provides a consistent interface
    for all supported Re-ID backbones.

    Outputs:
        - patch_tokens: (B, N, D) spatial patch features
        - cls_token: (B, D) global feature
    """

    def __init__(
        self,
        backbone_type: str,
        config: MemoryConfig,
    ) -> None:
        super().__init__()

        self.backbone_type = backbone_type
        self.config = config
        self.feat_dim = config.feat_dim
        self.spec = BACKBONE_REGISTRY.get(backbone_type)          # registry facts (None for "custom")

        # Build the appropriate backbone. BNNeck and memory are sized to its native dim, so there
        # is no dimension adapter.
        self.backbone = self._build_backbone()
        if self.feat_dim != config.feat_dim:
            raise ValueError(
                f"{backbone_type}: native feature dim {self.feat_dim} != config.feat_dim {config.feat_dim}; "
                "BNNeck and memory are sized to the native dim (see recognize.backbones)"
            )

    def _released_weights(self) -> Path:
        path = self.spec.weights_path() if self.spec is not None else None
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"{self.backbone_type}: released weights missing at {path}; run `python scripts/fetch_weights.py {self.backbone_type}`"
            )
        return path

    def _build_backbone(self) -> nn.Module:
        """Build backbone based on type."""
        if self.backbone_type == "transreid":
            return self._build_transreid()
        elif self.backbone_type == "magiv2":
            return self._build_magiv2()
        elif self.backbone_type == "magiv3":
            return self._build_magiv3()
        elif self.backbone_type == "instructreid":
            return self._build_instructreid()
        elif self.backbone_type == "reid5o":
            return self._build_reid5o()
        elif self.backbone_type == "custom":
            return self._build_custom()
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone_type}")

    def _build_transreid(self) -> nn.Module:
        """TransReID ViT-B/16.

        Architecture from timm, weights from the released TransReID Market-1501 checkpoint
        (key-mapped, strict), patch stride from the registry (12, as released), position grid
        taken from the checkpoint.
        """
        from timm.models.vision_transformer import vit_base_patch16_224

        backbone = vit_base_patch16_224(pretrained=False)
        backbone.reset_classifier(0)
        self.feat_dim = 768
        stride = self.spec.patch_stride or 16
        backbone.patch_embed.proj.stride = (stride, stride)
        path = self._released_weights()
        raw = torch.load(path, map_location="cpu", weights_only=False)
        raw = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) and "base.cls_token" not in raw else raw
        sd = map_transreid_keys(raw)
        gh, gw = infer_pos_grid(sd["pos_embed"].shape[1], (self.config.image_height, self.config.image_width), 16, stride)
        backbone.pos_embed = nn.Parameter(torch.zeros_like(sd["pos_embed"]))
        load_state_dict_checked(backbone, sd)
        self.pos_grid = (gh, gw)
        print(f"Loaded TransReID released weights {path.name}: {len(sd)} tensors, stride {stride}, pos grid {gh}x{gw}")
        return backbone

    def _build_magiv2(self) -> nn.Module:
        """MagiV2 crop encoder (ViT-MAE) at the pinned revision, masking off and asserted."""
        from transformers import AutoModel

        repo = self.spec.hf_repo if self.spec else "ragavsachdeva/magiv2"
        if self.config.backbone_config and self.config.backbone_config.get("model_name"):
            repo = self.config.backbone_config["model_name"]
        model = AutoModel.from_pretrained(repo, revision=self.spec.hf_revision if self.spec else None, trust_remote_code=True)
        backbone = model.crop_embedding_model if hasattr(model, "crop_embedding_model") else model
        self.feat_dim = getattr(backbone.config, "hidden_size", 768)
        found = disable_on_backbone(backbone)
        if mask_ratio_of(backbone) not in (None, 0.0):
            raise RuntimeError("MagiV2 crop encoder still masks patches")
        if found not in (None, 0.0):
            print(f"MagiV2: ViT-MAE mask_ratio {found} -> 0.0 (token order pinned)")
        return backbone

    def _build_magiv3(self) -> nn.Module:
        """Build MagiV3 backbone (Florence2-based)."""
        from transformers import AutoModelForCausalLM

        model_name = self.config.backbone_config.get("model_name", "ragavsachdeva/magiv3") if self.config.backbone_config else "ragavsachdeva/magiv3"

        if self.spec is not None and not (self.config.backbone_config and self.config.backbone_config.get("model_name")):
            model_name = self.spec.hf_repo
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=self.spec.hf_revision if self.spec else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,   # fp16 on GPU, fp32 on CPU
            trust_remote_code=True,
            attn_implementation="eager",
        )

        # Florence2 image features are 1024-d
        self.feat_dim = 1024
        return model

    def _build_instructreid(self) -> nn.Module:
        """Build InstructReID backbone."""
        project_root = Path(__file__).parent.parent.parent.parent
        instruct_path = project_root / "reid_models" / "Instruct-ReID-main"
        sys.path.insert(0, str(instruct_path))

        from reid.models.backbone.pass_vit import vit_base_patch16_224_TransReID

        backbone = vit_base_patch16_224_TransReID(
            img_size=(self.config.image_height, self.config.image_width),
            sie_xishu=3.0,
            camera=0,
            view=0,
            stride_size=[16, 16],
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            gem_pool=False,
            stem_conv=False,
        )

        self.feat_dim = 768
        path = self._released_weights()
        raw = torch.load(path, map_location="cpu", weights_only=False)
        sd = map_instructreid_keys(raw.get("state_dict", raw))
        load_state_dict_checked(backbone, sd)
        print(f"Loaded Instruct-ReID released weights {path.name}: {len(sd)} tensors (visual encoder)")
        return backbone

    def _build_reid5o(self) -> nn.Module:
        """Build ReID5o backbone (CLIP-based)."""
        project_root = Path(__file__).parent.parent.parent.parent
        reid5o_path = project_root / "reid_models" / "ReID5o"
        sys.path.insert(0, str(reid5o_path))

        from utils.iotools import load_train_configs
        from model import build_model

        config_path = self.config.backbone_config.get("config_path") if self.config.backbone_config else None
        if not config_path and self.spec is not None and self.spec.reid5o_config:
            config_path = str(project_root / self.spec.reid5o_config)
        if config_path:
            args = load_train_configs(config_path)
            args.training = False
            backbone = build_model(args, num_classes=1000)
        else:
            raise ValueError("ReID5o requires config_path in backbone_config")

        ckpt = self.config.backbone_checkpoint
        if not ckpt:
            ckpt = str(self._released_weights())
        if ckpt:
            self._load_checkpoint(backbone, ckpt)

        # ReID5o uses 512-dim CLIP embeddings (native dim)
        self.feat_dim = 512
        return backbone

    def _build_custom(self) -> nn.Module:
        """Build custom backbone from config."""
        if not self.config.backbone_config or "module" not in self.config.backbone_config:
            raise ValueError("Custom backbone requires 'module' and 'class' in backbone_config")

        import importlib
        module = importlib.import_module(self.config.backbone_config["module"])
        cls = getattr(module, self.config.backbone_config["class"])
        kwargs = self.config.backbone_config.get("kwargs", {})
        return cls(**kwargs)

    def _load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        prefix: str = "",
    ) -> None:
        """Load checkpoint into model."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        # Strip an optional wrapper prefix ("backbone.", "module.") only if the
        # checkpoint actually uses it: filtering on a prefix that does not occur
        # would drop every key, and a non-strict load would then load nothing.
        if prefix and any(k.startswith(prefix) for k in state_dict):
            state_dict = {
                k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
            }
        if not state_dict:
            raise RuntimeError(f"checkpoint {ckpt_path} yielded no keys (prefix={prefix!r})")

        # Heads that a backbone-only checkpoint legitimately lacks. Everything
        # else must match exactly; shape mismatches raise.
        load_state_dict_checked(
            model, state_dict, allow_missing_prefixes=("fc.", "head.", "classifier."),
        )
        n_loaded = sum(1 for k in state_dict if k in model.state_dict())
        print(f"Loaded checkpoint from {ckpt_path} ({n_loaded}/{len(model.state_dict())} keys)")

    @staticmethod
    def _interpolate_pos_embed(
        pos_embed: torch.Tensor,
        src_shape: Tuple[int, int],
        dst_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Interpolate 2D positional embeddings from src grid to dst grid."""
        cls_tok, grid = pos_embed[:, :1], pos_embed[:, 1:]
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape
        grid = grid.reshape(1, src_h, src_w, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(dst_h, dst_w), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, dst_h * dst_w, -1)
        return torch.cat([cls_tok, grid], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from backbone.

        Args:
            x: (B, C, H, W) input images

        Returns:
            patch_tokens: (B, N, D) patch features
            cls_token: (B, D) global feature
        """
        if self.backbone_type == "transreid":
            return self._forward_transreid(x)
        elif self.backbone_type == "magiv2":
            return self._forward_magiv2(x)
        elif self.backbone_type == "magiv3":
            return self._forward_magiv3(x)
        elif self.backbone_type == "instructreid":
            return self._forward_instructreid(x)
        elif self.backbone_type == "reid5o":
            return self._forward_reid5o(x)
        else:
            return self._forward_generic(x)

    def _forward_transreid(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """TransReID forward: patch conv at the registry stride, position grid from the loaded weights."""
        b = x.shape[0]
        backbone = self.backbone  # raw timm VisionTransformer

        tokens = backbone.patch_embed.proj(x)              # (B, D, gh, gw); stride set at build time
        gh, gw = int(tokens.shape[-2]), int(tokens.shape[-1])
        tokens = backbone.patch_embed.norm(tokens.flatten(2).transpose(1, 2))

        cls_tokens = backbone.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        src_grid = tuple(getattr(self, "pos_grid", (14, 14)))
        if (gh, gw) != src_grid:
            pos_embed = self._interpolate_pos_embed(backbone.pos_embed, src_grid, (gh, gw))
        else:
            pos_embed = backbone.pos_embed
        tokens = backbone.pos_drop(tokens + pos_embed)

        for blk in backbone.blocks:
            tokens = blk(tokens)
        tokens = backbone.norm(tokens)

        return tokens[:, 1:], tokens[:, 0]

    def _forward_magiv2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MagiV2 forward pass."""
        outputs = self.backbone(x)

        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            tokens = outputs

        if tokens.dim() == 3:
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            cls_token = tokens
            patch_tokens = tokens.unsqueeze(1)

        return patch_tokens, cls_token

    def _forward_magiv3(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MagiV3 (Florence2) forward pass.

        Note: MagiV3 uses square 384x384 images. The patch tokens may not
        have the same spatial structure as person Re-ID models. Part attention
        should be disabled or will use estimated grid size.
        """
        # Match the encoder dtype (fp16 on GPU, fp32 on CPU)
        x_half = x.to(next(self.backbone.parameters()).dtype)

        if hasattr(self.backbone, "_encode_image"):
            result = self.backbone._encode_image(x_half)
            image_features = result[0] if isinstance(result, tuple) else result
            # Florence2 returns (B, num_patches, dim) where num_patches varies
            cls_token = image_features.mean(dim=1)
            patch_tokens = image_features
        else:
            # Fallback
            cls_token = self.backbone(x_half)
            patch_tokens = cls_token.unsqueeze(1)

        # Convert to float32 for downstream modules
        cls_token = cls_token.float()
        patch_tokens = patch_tokens.float()

        return patch_tokens, cls_token

    def _forward_instructreid(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """InstructReID forward pass."""
        output = self.backbone(x)

        if isinstance(output, tuple):
            cls_token = output[0]
            # Try to get patch tokens from token output
            if len(output) > 2:
                patch_tokens = output[2]  # token features
            else:
                patch_tokens = cls_token.unsqueeze(1)
        else:
            if output.dim() == 3:
                cls_token = output[:, 0]
                patch_tokens = output[:, 1:]
            else:
                cls_token = output
                patch_tokens = output.unsqueeze(1)

        return patch_tokens, cls_token

    def _forward_reid5o(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ReID5o forward pass."""
        cls_token = self.backbone.encode_rgb_cls(x)

        # ReID5o doesn't provide patch tokens directly
        # We'll use the cls token repeated
        patch_tokens = cls_token.unsqueeze(1)

        return patch_tokens, cls_token

    def _forward_generic(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generic forward pass for custom backbones."""
        output = self.backbone(x)

        if isinstance(output, tuple):
            cls_token = output[0]
            patch_tokens = output[1] if len(output) > 1 else cls_token.unsqueeze(1)
        elif output.dim() == 3:
            cls_token = output[:, 0]
            patch_tokens = output[:, 1:]
        else:
            cls_token = output
            patch_tokens = output.unsqueeze(1)

        return patch_tokens, cls_token


class MemoryEnhancedReID(nn.Module):
    """
    Universal Memory-Enhanced Re-ID Model.

    Works with any supported backbone:
    - TransReID (ViT-B/16, 768-dim)
    - MagiV2 (ViT-based, 768-dim)
    - MagiV3 (Florence2, 1024-dim)
    - InstructReID (ViT-B/16, 768-dim)
    - ReID5o (CLIP ViT-B/16, 512-dim)

    Args:
        config: MemoryConfig with all hyperparameters
    """

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()

        self.config = config

        # Build backbone wrapper
        self.backbone_wrapper = BackboneWrapper(
            backbone_type=config.backbone_type,
            config=config,
        )

        # Native dims: the wrapper has already asserted feat_dim == config.feat_dim
        actual_feat_dim = config.feat_dim

        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.backbone_wrapper.backbone.parameters():
                param.requires_grad = False
            self.backbone_wrapper.backbone.eval()
            print(f"Backbone frozen ({config.backbone_type})")

        # Inject LoRA AFTER freezing (LoRA params are created unfrozen)
        if config.use_lora:
            from .lora import inject_lora
            # The probe runs one image through the wrapper's own forward, so adapters land only on
            # modules the image path calls. Florence-2's language decoder and ReID5o's text
            # encoder never run on an image, so adapters there could change nothing.
            probe = torch.zeros(1, 3, config.image_height, config.image_width)
            injected = inject_lora(
                self.backbone_wrapper.backbone,
                backbone_type=config.backbone_type,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                num_layers=config.lora_layers,
                probe_forward=lambda: self.backbone_wrapper(probe),
            )
            if not injected:
                raise RuntimeError(f"LoRA requested but no adapter was placed on {config.backbone_type}'s image path")

        # Memory block
        if config.use_working_memory or config.use_episodic_memory:
            self.memory_block = MemoryBlock(
                num_characters=config.num_classes,
                working_capacity=config.working_capacity,
                slots_per_char=config.slots_per_char,
                feat_dim=config.feat_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                episodic_id_drop_rate=config.episodic_id_drop_rate,
                use_working_memory=config.use_working_memory,
                use_episodic_memory=config.use_episodic_memory,
                residual_max_ratio=config.residual_max_ratio,
            )
        else:
            self.memory_block = None

        # BNNeck for Re-ID
        self.bnneck = BNNeck(config.feat_dim)

        # Classifier
        self.classifier = nn.Linear(config.feat_dim, config.num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def train(self, mode: bool = True):
        """Override to keep backbone in eval mode (except LoRA modules)."""
        super().train(mode)
        if self.config.freeze_backbone and hasattr(self.backbone_wrapper, 'backbone'):
            self.backbone_wrapper.backbone.eval()
            # Re-enable train mode for LoRA modules (needed for dropout)
            if self.config.use_lora:
                from .lora import LoRALinear
                for m in self.backbone_wrapper.backbone.modules():
                    if isinstance(m, LoRALinear):
                        m.train(mode)
        return self

    def forward(
        self,
        x: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        update_working: bool = True,
        update_episodic: bool = False,
        episodic_search_all: bool = False,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input images
            char_ids: (B,) character IDs (for episodic memory)
            use_memory: If True, use memory modules
            update_working: If True, update working memory
            update_episodic: If True, update episodic memory prototypes
            return_all: If True, return all intermediate features

        Returns:
            dict with logits, bn_feat, and optionally more
        """
        B = x.shape[0]

        # Extract backbone features (frozen, but enable grad if LoRA is active)
        use_no_grad = self.config.freeze_backbone and not self.config.use_lora
        with torch.no_grad() if use_no_grad else torch.enable_grad():
            _, cls_token = self.backbone_wrapper(x)

        # BNNeck BEFORE memory so cross-attention query and stored prototypes
        # are in the same BN-normalized space.
        pre_memory_bn = self.bnneck(cls_token)

        # Memory block (operates in BN-normalized space)
        if self.memory_block is not None and use_memory:
            memory_out = self.memory_block(
                pre_memory_bn,
                char_ids=char_ids,
                update_working=update_working,
                update_episodic=update_episodic,
                episodic_search_all=episodic_search_all,
            )
            final_feat = memory_out["output"]
            gate_weights = memory_out["gate_weights"]
        else:
            final_feat = pre_memory_bn
            gate_weights = None

        # Re-normalize after memory (residual connections break L2 norm)
        bn_feat = F.normalize(final_feat, p=2, dim=1)

        # Classifier
        logits = self.classifier(bn_feat)

        # Build output dict
        output = {
            "bn_feat": bn_feat,
            "logits": logits,
            "final_feat": final_feat,
            "pre_memory_feat": pre_memory_bn,
        }

        if return_all:
            output.update({
                "cls_token": cls_token,
                "gate_weights": gate_weights,
            })

        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized features for inference (no memory update).

        Returns bn_feat, which is already L2-normalized by BNNeck.
        """
        output = self.forward(
            x,
            char_ids=None,
            use_memory=True,
            update_working=False,
            update_episodic=False,
        )
        return output["bn_feat"]

    def reset_working_memory(self) -> None:
        """Reset working memory (call at start of new page/chapter)."""
        if self.memory_block is not None:
            self.memory_block.reset_working_memory()

    def reset_all_memory(self) -> None:
        """Reset both working and episodic memory."""
        if self.memory_block is not None:
            self.memory_block.reset()

    def reinit_for_open_set(
        self,
        num_characters: int,
        support_features: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Reinitialize memory for open-set inference with a new set of characters.

        This preserves the learned attention weights while creating a fresh
        prototype bank for the new characters.

        Args:
            num_characters: Number of characters in the new set
            support_features: (N, D) features from first instances (optional)
            support_labels: (N,) character IDs for support features (optional)

        Example:
            # Load a trained model
            model = recognize.checkpoints.load_checkpoint("model.pth").model

            # Prepare for new comic with 15 characters
            model.reinit_for_open_set(num_characters=15)

            # Initialize from first instances
            first_instances = ...  # (15, C, H, W)
            with torch.no_grad():
                features = model.extract_features(first_instances)
            model.memory_block.initialize_from_support(
                features,
                char_ids=torch.arange(15),
                method="diverse"
            )

            # Now ready for inference on new panels
        """
        if self.memory_block is None:
            return

        from .memory_modules import MemoryBlock

        # Get current config
        old_block = self.memory_block
        device = next(old_block.parameters()).device

        # Create new memory block with correct num_characters
        new_block = MemoryBlock(
            num_characters=num_characters,
            working_capacity=old_block.working_memory.capacity,
            slots_per_char=old_block.episodic_memory.slots_per_char,
            feat_dim=old_block.episodic_memory.feat_dim,
            num_heads=old_block.working_memory.cross_attention.num_heads,
            dropout=0.1,  # Default
            episodic_id_drop_rate=old_block.episodic_id_drop_rate,
            use_working_memory=old_block.use_working_memory,
            use_episodic_memory=old_block.use_episodic_memory,
            residual_max_ratio=old_block.fusion.residual_max_ratio,
        ).to(device)

        # Copy LEARNED weights only (attention, projections, gate).
        # Buffers (memory_bank, prototypes, etc.) are NOT copied: they
        # have different shapes for the new character count and start fresh.

        # Working memory: learned params only (buffers are per-character, new shape)
        new_block.working_memory.cross_attention.load_state_dict(
            old_block.working_memory.cross_attention.state_dict()
        )
        new_block.working_memory.query_norm.load_state_dict(
            old_block.working_memory.query_norm.state_dict()
        )
        new_block.working_memory.memory_norm.load_state_dict(
            old_block.working_memory.memory_norm.state_dict()
        )
        new_block.working_memory.output_norm.load_state_dict(
            old_block.working_memory.output_norm.state_dict()
        )
        new_block.working_memory.output_proj.load_state_dict(
            old_block.working_memory.output_proj.state_dict()
        )

        # Episodic memory: learned params only (prototypes are non-parametric buffers)
        new_block.episodic_memory.cross_attention.load_state_dict(
            old_block.episodic_memory.cross_attention.state_dict()
        )
        new_block.episodic_memory.query_norm.load_state_dict(
            old_block.episodic_memory.query_norm.state_dict()
        )
        new_block.episodic_memory.memory_norm.load_state_dict(
            old_block.episodic_memory.memory_norm.state_dict()
        )
        new_block.episodic_memory.output_norm.load_state_dict(
            old_block.episodic_memory.output_norm.state_dict()
        )
        new_block.episodic_memory.output_proj.load_state_dict(
            old_block.episodic_memory.output_proj.state_dict()
        )

        # Gated fusion (all learned, no buffers)
        new_block.fusion.load_state_dict(old_block.fusion.state_dict())

        # Replace memory block, then propagate the current model's train/eval mode.
        # nn.Module objects are in train mode by default; without this, every
        # evaluation function that calls reinit_for_open_set() would run forward
        # passes with dropout *active*, introducing randomness and degrading metrics.
        self.memory_block = new_block
        new_block.train(self.training)

        # Initialize from support if provided
        if support_features is not None:
            self.memory_block.initialize_from_support(
                support_features,
                char_ids=support_labels,
                method="diverse",
            )

    def initialize_from_gallery(
        self,
        gallery_images: torch.Tensor,
        gallery_labels: torch.Tensor,
    ) -> None:
        """
        Initialize episodic memory from a gallery of first instances.

        Single-pass: extract clean (no-memory) bn_feat → prototypes + WM seed.
        Prototypes stay in BN-normalized space, matching the training protocol
        where episodic memory is always initialized from clean features.

        At query time (identify_character), memory-enhanced bn_feat is compared
        against these clean prototypes: the memory module's job is to refine
        features toward the correct prototype direction.

        Args:
            gallery_images: (N, C, H, W) images of first instances
            gallery_labels: (N,) character IDs (0 to num_chars-1)
        """
        if self.memory_block is None:
            raise RuntimeError("Model has no memory block")

        # Reinit for correct number of characters
        num_chars = gallery_labels.max().item() + 1
        self.reinit_for_open_set(num_chars)

        self.eval()
        with torch.no_grad():
            output = self.forward(
                gallery_images,
                use_memory=False,
                return_all=True,
            )
            bn_feat = output["bn_feat"]

            # Initialize episodic prototypes and seed working memory
            self.memory_block.episodic_memory.initialize_from_support(
                bn_feat, char_ids=gallery_labels, method="diverse"
            )
            self.memory_block.working_memory.update(bn_feat, gallery_labels)

    def identify_character(
        self,
        query_images: torch.Tensor,
        update_memory: bool = True,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Identify characters in query images using initialized memory.

        Args:
            query_images: (B, C, H, W) query images
            update_memory: If True, update working memory (for sequential processing)
            return_features: If True, include extracted features in output

        Returns:
            dict with:
                - similarities: (B, num_chars) cosine similarity to each character
                - predictions: (B,) predicted character IDs
                - confidences: (B,) confidence scores (max similarity)
                - features: (B, D) extracted features (if return_features=True)
        """
        if self.memory_block is None:
            raise RuntimeError("Model has no memory block")

        self.eval()
        with torch.no_grad():
            # Pass 1: episodic memory only (char_ids=None) → rough prediction
            output1 = self.forward(
                query_images,
                char_ids=None,
                use_memory=True,
                update_working=False,
                update_episodic=False,
                return_all=True,
            )
            rough_similarities = self.memory_block.episodic_memory.compute_similarity(
                output1["bn_feat"]
            )
            _, rough_pred = rough_similarities.max(dim=1)

            # Pass 2: route working memory with rough prediction → refined feature
            output2 = self.forward(
                query_images,
                char_ids=rough_pred,
                use_memory=True,
                update_working=False,
                update_episodic=False,
                episodic_search_all=True,
                return_all=True,
            )

            feat_bn = output2["bn_feat"]            # refined; for prototype matching
            pre_mem = output2["pre_memory_feat"]     # BN-normalized; for working memory

            # Final similarity using refined features
            similarities = self.memory_block.episodic_memory.compute_similarity(feat_bn)
            confidences, predictions = similarities.max(dim=1)

            # Update working memory (BN-normalized space, matching memory block input)
            if update_memory:
                self.memory_block.working_memory.update(pre_mem, predictions)

        result = {
            "similarities": similarities,
            "predictions": predictions,
            "confidences": confidences,
        }

        if return_features:
            result["features"] = output2["bn_feat"]

        return result

    def get_trainable_param_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_param_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
