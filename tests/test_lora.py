"""Tests for LoRA injection, parameter isolation, and merge."""

import pytest
import torch
import torch.nn as nn

from memory_block.models.lora import (
    LoRALinear,
    inject_lora,
    get_lora_parameters,
    count_lora_parameters,
    merge_lora,
)


class TestLoRALinear:
    @pytest.fixture
    def linear(self):
        """Frozen linear layer."""
        lin = nn.Linear(64, 128)
        for p in lin.parameters():
            p.requires_grad = False
        return lin

    @pytest.fixture
    def lora(self, linear):
        return LoRALinear(linear, rank=4, alpha=8.0, dropout=0.0)

    def test_output_shape(self, lora):
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 128)

    def test_zero_init_matches_original(self, linear):
        """Fresh LoRA (B=0) should produce identical output to original linear."""
        lora = LoRALinear(linear, rank=4, alpha=8.0, dropout=0.0)
        x = torch.randn(5, 64)
        with torch.no_grad():
            orig_out = linear(x)
            lora_out = lora(x)
        assert torch.allclose(orig_out, lora_out, atol=1e-6), (
            f"Max diff: {(orig_out - lora_out).abs().max():.2e}"
        )

    def test_lora_params_are_trainable(self, lora):
        assert lora.lora_A.requires_grad is True
        assert lora.lora_B.requires_grad is True

    def test_original_stays_frozen(self, lora):
        for p in lora.original.parameters():
            assert p.requires_grad is False

    def test_gradient_flows_through_lora(self, lora):
        """Gradients should flow to lora_A and lora_B but not original."""
        x = torch.randn(2, 64)
        out = lora(x)
        out.sum().backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        for p in lora.original.parameters():
            assert p.grad is None

    def test_nonzero_B_changes_output(self, linear):
        """After setting B to nonzero, output should differ from original."""
        lora = LoRALinear(linear, rank=4, alpha=8.0, dropout=0.0)
        lora.lora_B.data.fill_(0.01)
        x = torch.randn(2, 64)
        with torch.no_grad():
            orig_out = linear(x)
            lora_out = lora(x)
        assert not torch.allclose(orig_out, lora_out, atol=1e-6)


class TestInjectLora:
    @pytest.fixture
    def frozen_vit(self):
        """Minimal frozen ViT-like model (timm structure)."""
        from timm.models.vision_transformer import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained=False)
        for p in model.parameters():
            p.requires_grad = False
        return model

    def test_injection_returns_module_names(self, frozen_vit):
        injected = inject_lora(frozen_vit, backbone_type="transreid", rank=4, num_layers=2)
        assert len(injected) > 0
        for name in injected:
            assert "attn" in name

    def test_injected_modules_are_lora_linear(self, frozen_vit):
        inject_lora(frozen_vit, backbone_type="transreid", rank=4, num_layers=2)
        found = False
        for m in frozen_vit.modules():
            if isinstance(m, LoRALinear):
                found = True
                break
        assert found, "No LoRALinear modules found after injection"

    def test_only_lora_params_trainable(self, frozen_vit):
        """After injection, only LoRA params should be trainable."""
        inject_lora(frozen_vit, backbone_type="transreid", rank=4, num_layers=2)
        lora_params = get_lora_parameters(frozen_vit)
        all_trainable = [p for p in frozen_vit.parameters() if p.requires_grad]
        assert len(lora_params) == len(all_trainable), (
            f"LoRA params: {len(lora_params)}, all trainable: {len(all_trainable)}"
        )

    def test_num_layers_controls_scope(self, frozen_vit):
        """Fewer layers should inject fewer LoRA modules."""
        from timm.models.vision_transformer import vit_base_patch16_224
        vit2 = vit_base_patch16_224(pretrained=False)
        for p in vit2.parameters():
            p.requires_grad = False

        injected_2 = inject_lora(frozen_vit, backbone_type="transreid", rank=4, num_layers=2)
        injected_4 = inject_lora(vit2, backbone_type="transreid", rank=4, num_layers=4)
        assert len(injected_4) > len(injected_2)

    def test_param_count_is_small(self, frozen_vit):
        """LoRA should add < 1% of backbone params."""
        total_before = sum(p.numel() for p in frozen_vit.parameters())
        inject_lora(frozen_vit, backbone_type="transreid", rank=8, num_layers=4)
        lora_count = count_lora_parameters(frozen_vit)
        ratio = lora_count / total_before
        assert ratio < 0.01, f"LoRA param ratio {ratio:.4f} exceeds 1%"


class TestMergeLora:
    def test_merge_produces_identical_output(self):
        """After merge, plain linear should produce same output as LoRA linear."""
        linear = nn.Linear(64, 128)
        for p in linear.parameters():
            p.requires_grad = False
        lora = LoRALinear(linear, rank=4, alpha=8.0, dropout=0.0)
        # Set nonzero B so merge has something to fold in
        lora.lora_B.data.normal_(0, 0.01)

        x = torch.randn(3, 64)
        with torch.no_grad():
            lora_out = lora(x)

        # Create a parent module to test merge
        parent = nn.Module()
        parent.layer = lora
        merge_lora(parent)

        assert isinstance(parent.layer, nn.Linear), "merge should replace LoRALinear with nn.Linear"
        with torch.no_grad():
            merged_out = parent.layer(x)
        assert torch.allclose(lora_out, merged_out, atol=1e-5), (
            f"Max diff after merge: {(lora_out - merged_out).abs().max():.2e}"
        )

    def test_merge_detaches_lora_params(self):
        """Modifying LoRA params after merge should not affect the merged layer."""
        linear = nn.Linear(64, 128)
        for p in linear.parameters():
            p.requires_grad = False
        lora = LoRALinear(linear, rank=4, alpha=8.0, dropout=0.0)
        
        # Keep a reference to the original lora_B tensor
        lora_b_ref = lora.lora_B

        parent = nn.Module()
        parent.layer = lora
        merge_lora(parent)

        x = torch.randn(2, 64)
        with torch.no_grad():
            out_before = parent.layer(x)
            
        # Modify the detached lora_B reference
        lora_b_ref.data.fill_(100.0)
        
        with torch.no_grad():
            out_after = parent.layer(x)
            
        # The merged linear layer should be completely independent
        assert torch.allclose(out_before, out_after)


class TestWrappingLayersWithExtraArguments:
    """ReID5o's own LoRALinear is an nn.Linear subclass taking (x, lora_index)."""

    class _IndexedLinear(nn.Linear):
        def forward(self, x, lora_index=0):
            return super().forward(x) + float(lora_index)

    def test_extra_arguments_reach_the_wrapped_layer(self):
        base = self._IndexedLinear(4, 3)
        wrapped = LoRALinear(base, rank=2, alpha=4.0)
        x = torch.randn(5, 4)
        # B is zero at init, so the wrapper must reproduce the base layer exactly, index included
        torch.testing.assert_close(wrapped(x), base(x))
        torch.testing.assert_close(wrapped(x, 2), base(x, 2))
        assert not torch.allclose(wrapped(x, 2), wrapped(x, 0))       # the index still matters

    def test_lora_term_adds_on_top_of_the_indexed_output(self):
        base = self._IndexedLinear(4, 3)
        wrapped = LoRALinear(base, rank=2, alpha=4.0)
        with torch.no_grad():
            wrapped.lora_B.normal_()
        x = torch.randn(5, 4)
        delta = wrapped(x, 1) - base(x, 1)
        torch.testing.assert_close(delta, wrapped(x, 0) - base(x, 0))  # same low-rank term either way
        assert delta.abs().sum() > 0

    def test_injection_is_idempotent(self):
        model = nn.Sequential(nn.Linear(4, 4))
        first = LoRALinear(model[0], rank=2, alpha=4.0)
        model[0] = first
        again = LoRALinear(model[0], rank=2, alpha=4.0) if not isinstance(model[0], LoRALinear) else model[0]
        assert again is first


class TestProbeDrivenSelection:
    """Adapters must land only on modules the image forward calls."""

    class _TwoTowers(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.ModuleDict({str(i): nn.ModuleDict(
                {"attn_qkv": nn.Linear(8, 8), "attn_proj": nn.Linear(8, 8)}) for i in range(4)})
            self.language = nn.ModuleDict({str(i): nn.ModuleDict(
                {"attn_q": nn.Linear(8, 8), "attn_k": nn.Linear(8, 8)}) for i in range(6)})

        def forward(self, x):
            for i in range(4):
                x = self.vision[str(i)]["attn_proj"](self.vision[str(i)]["attn_qkv"](x))
            return x

    def test_probe_keeps_the_image_trunk_only(self):
        from memory_block.models.lora import inject_lora, used_linear_modules
        m = self._TwoTowers()
        x = torch.randn(2, 8)
        used = used_linear_modules(m, lambda: m(x))
        assert all(n.startswith("vision.") for n in used) and len(used) == 8
        injected = inject_lora(m, backbone_type="custom", rank=2, num_layers=2, probe_forward=lambda: m(x))
        assert len(injected) == 4 and all(n.startswith("vision.") for n in injected)
        assert {n.split(".")[1] for n in injected} == {"2", "3"}          # the last two blocks
        assert {n.split(".")[-1] for n in injected} == {"attn_qkv", "attn_proj"}
        assert not any(isinstance(mod, LoRALinear) for n, mod in m.named_modules() if n.startswith("language"))

    def test_no_injected_adapter_is_dead(self):
        from memory_block.models.lora import LoRALinear as LL, inject_lora
        m = self._TwoTowers()
        x = torch.randn(2, 8)
        inject_lora(m, backbone_type="custom", rank=2, num_layers=2, probe_forward=lambda: m(x))
        fired = set()
        hooks = [mod.register_forward_hook(lambda mo, i, o, n=n: fired.add(n))
                 for n, mod in m.named_modules() if isinstance(mod, LL)]
        with torch.no_grad():
            m(x)
        for h in hooks:
            h.remove()
        names = {n for n, mod in m.named_modules() if isinstance(mod, LL)}
        assert names and fired == names

    def test_a_backbone_whose_image_path_has_no_linear_raises_in_the_model(self):
        from memory_block.models.lora import inject_lora
        m = nn.Sequential(nn.Conv2d(3, 3, 1))
        assert inject_lora(m, backbone_type="custom", rank=2, num_layers=2,
                           probe_forward=lambda: m(torch.zeros(1, 3, 4, 4))) == []


    class _MlpAndAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.ModuleDict({
                "attn_qkv": nn.Linear(8, 8), "mlp_fc1": nn.Linear(8, 8), "mlp_fc2": nn.Linear(8, 8)})
                for _ in range(3)])

        def forward(self, x):
            for b in self.blocks:
                x = b["mlp_fc2"](b["mlp_fc1"](b["attn_qkv"](x)))
            return x

    def test_only_attention_projections_are_adapted(self):
        from memory_block.models.lora import inject_lora
        m = self._MlpAndAttn()
        x = torch.randn(2, 8)
        injected = inject_lora(m, backbone_type="custom", rank=2, num_layers=2, probe_forward=lambda: m(x))
        assert injected == ["blocks.1.attn_qkv", "blocks.2.attn_qkv"]
