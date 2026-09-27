"""Tests for training CLI argument parsing and defaults."""

import sys
import pytest


def _parse(argv):
    """Parse CLI args without running training."""
    original = sys.argv
    sys.argv = ["train.py"] + argv
    try:
        from memory_block.training.train import parse_args
        return parse_args()
    finally:
        sys.argv = original

BACKBONES = ["transreid", "instructreid", "magiv2", "magiv3", "reid5o"]



class TestCLIDefaults:
    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_ce_weight_default(self, backbone):
        args = _parse(["--data-dir", "/tmp", "--backbone", backbone])
        assert args.ce_weight == 0.3

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_prototype_weight_default(self, backbone):
        args = _parse(["--data-dir", "/tmp", "--backbone", backbone])
        assert args.prototype_weight == 1.0

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_episodic_id_drop_rate_default(self, backbone):
        args = _parse(["--data-dir", "/tmp", "--backbone", backbone])
        assert args.episodic_id_drop_rate == 0.5

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_lora_defaults(self, backbone):
        args = _parse(["--data-dir", "/tmp", "--backbone", backbone])
        assert args.use_lora is False
        assert args.lora_rank == 8
        assert args.lora_alpha == 16.0
        assert args.lora_layers == 4
        assert args.lora_lr == 1e-5


class TestEpisodicTrainingDefault:
    """Memory ON always implies episodic training. No-memory disables it."""

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_memory_enabled_implies_episodic(self, backbone):
        """Default (no --no-memory) → episodic training ON."""
        args = _parse(["--data-dir", "/tmp", "--backbone", backbone])
        memory_enabled = not args.no_memory
        assert memory_enabled is True

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_no_memory_disables_episodic(self, backbone):
        """--no-memory → baseline, no episodic."""
        args = _parse(["--data-dir", "/tmp", "--backbone", backbone, "--no-memory"])
        memory_enabled = not args.no_memory
        assert memory_enabled is False


class TestLoRACLI:
    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_use_lora_flag(self, backbone):
        args = _parse([
            "--data-dir", "/tmp", "--backbone", backbone, "--use-lora",
        ])
        assert args.use_lora is True

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_lora_rank_override(self, backbone):
        args = _parse([
            "--data-dir", "/tmp", "--backbone", backbone,
            "--use-lora", "--lora-rank", "16",
        ])
        assert args.lora_rank == 16

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_lora_layers_override(self, backbone):
        args = _parse([
            "--data-dir", "/tmp", "--backbone", backbone,
            "--use-lora", "--lora-layers", "6",
        ])
        assert args.lora_layers == 6
