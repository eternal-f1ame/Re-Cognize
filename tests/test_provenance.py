"""Tests for recognize.provenance and recognize.protocol_constants."""

import argparse
import dataclasses
import hashlib
import subprocess

import pytest
import torch

from recognize import protocol_constants
from recognize.protocol_constants import B_MAX, EVAL_SEEDS
from recognize.provenance import assert_hashseed_pinned, stamp


@dataclasses.dataclass
class DummyConfig:
    """Module-level dataclass so torch.save can pickle it (see test below)."""

    epochs: int
    lr: float


EXPECTED_STAMP_KEYS = {
    "git_commit",
    "git_dirty",
    "args",
    "seeds",
    "pythonhashseed",
    "mask_ratio",
    "b_max",
    "split",
    "n_crops",
    "n_identities",
    "checkpoint_sha256",
    "checkpoint_config",
    "submit_commit",
    "git_dirty_files",
}


class TestProtocolConstants:
    def test_values(self):
        assert protocol_constants.TAU_NOV == 0.55
        assert protocol_constants.B_MAX == 50
        assert protocol_constants.GALLERY_RATIO == 0.2
        assert protocol_constants.EVAL_SEEDS == (0, 1, 2, 3, 4)
        assert protocol_constants.CROP_PADDING == 0.10
        assert protocol_constants.K_RANGE == (1, 2, 3, 4, 5)


class TestStampShape:
    def test_contains_every_key(self):
        result = stamp({}, None)
        assert set(result.keys()) == EXPECTED_STAMP_KEYS

    def test_git_dirty_is_bool(self):
        result = stamp({}, None)
        assert isinstance(result["git_dirty"], bool)

    def test_git_commit_is_str_or_none(self):
        result = stamp({}, None)
        assert result["git_commit"] is None or (
            isinstance(result["git_commit"], str) and len(result["git_commit"]) > 0
        )

    def test_accepts_namespace(self):
        ns = argparse.Namespace(foo="bar", seeds=[9])
        result = stamp(ns, None)
        assert result["args"] == {"foo": "bar", "seeds": [9]}
        assert result["seeds"] == [9]

    def test_accepts_plain_dict(self):
        result = stamp({"foo": "bar"}, None)
        assert result["args"] == {"foo": "bar"}


class TestStampDefaults:
    def test_seeds_default_to_eval_seeds(self):
        result = stamp({}, None)
        assert result["seeds"] == list(EVAL_SEEDS)

    def test_seeds_override(self):
        result = stamp({"seeds": [7, 8]}, None)
        assert result["seeds"] == [7, 8]

    def test_b_max_default(self):
        result = stamp({}, None)
        assert result["b_max"] == B_MAX

    def test_b_max_override(self):
        result = stamp({"b_max": 10}, None)
        assert result["b_max"] == 10

    def test_optional_fields_default_none(self):
        result = stamp({}, None)
        assert result["mask_ratio"] is None
        assert result["split"] is None
        assert result["n_crops"] is None
        assert result["n_identities"] is None

    def test_optional_fields_present(self):
        result = stamp(
            {
                "mask_ratio": 0.3,
                "split": "val",
                "n_crops": 123,
                "n_identities": 45,
            },
            None,
        )
        assert result["mask_ratio"] == 0.3
        assert result["split"] == "val"
        assert result["n_crops"] == 123
        assert result["n_identities"] == 45


class TestStampCheckpoint:
    def test_checkpoint_none_yields_none_fields(self):
        result = stamp({}, None)
        assert result["checkpoint_sha256"] is None
        assert result["checkpoint_config"] is None

    def test_checkpoint_sha256_matches_manual_digest(self, tmp_path):
        ckpt_path = tmp_path / "model.pth"
        torch.save({"state_dict": {}, "config": {"epochs": 5}}, ckpt_path)

        expected = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                expected.update(chunk)

        result = stamp({}, ckpt_path)
        assert result["checkpoint_sha256"] == expected.hexdigest()

    def test_checkpoint_config_dict_roundtrips(self, tmp_path):
        ckpt_path = tmp_path / "model.pth"
        config = {"epochs": 5, "lr": 1e-4, "backbone": "transreid"}
        torch.save({"state_dict": {}, "config": config}, ckpt_path)

        result = stamp({}, ckpt_path)
        assert result["checkpoint_config"] == config

    def test_checkpoint_config_dataclass_like_converted_via_vars(self, tmp_path):
        ckpt_path = tmp_path / "model.pth"
        cfg = DummyConfig(epochs=10, lr=1e-3)
        torch.save({"state_dict": {}, "config": cfg}, ckpt_path)

        result = stamp({}, ckpt_path)
        assert result["checkpoint_config"] == vars(cfg)

    def test_checkpoint_without_config_key_yields_none_config(self, tmp_path):
        ckpt_path = tmp_path / "model.pth"
        torch.save({"state_dict": {}}, ckpt_path)

        result = stamp({}, ckpt_path)
        assert result["checkpoint_config"] is None
        assert result["checkpoint_sha256"] is not None


class TestGitUnavailable:
    def test_git_missing_yields_none_commit_and_dirty(self, monkeypatch):
        import recognize.provenance as provenance_module

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("git not installed")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(provenance_module, "subprocess", subprocess)

        result = stamp({}, None)
        assert result["git_commit"] is None
        assert result["git_dirty"] is None


class TestAssertHashseedPinned:
    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
            assert_hashseed_pinned()

    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setenv("PYTHONHASHSEED", "")
        with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
            assert_hashseed_pinned()

    def test_does_not_raise_when_set(self, monkeypatch):
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        assert_hashseed_pinned()  # should not raise

    def test_stamp_reflects_pythonhashseed_env(self, monkeypatch):
        monkeypatch.setenv("PYTHONHASHSEED", "42")
        result = stamp({}, None)
        assert result["pythonhashseed"] == "42"

    def test_stamp_pythonhashseed_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        result = stamp({}, None)
        assert result["pythonhashseed"] is None


class TestSubmitCommit:
    """A campaign is pinned by the commit it was launched from, not by HEAD when a task starts."""

    def test_recorded_when_the_launcher_sets_it(self, monkeypatch):
        from recognize.provenance import stamp
        monkeypatch.setenv("SUBMIT_COMMIT", "abc1234")
        assert stamp({"x": 1}, None)["submit_commit"] == "abc1234"

    def test_absent_outside_a_campaign(self, monkeypatch):
        from recognize.provenance import stamp
        monkeypatch.delenv("SUBMIT_COMMIT", raising=False)
        assert stamp({"x": 1}, None)["submit_commit"] is None


class TestDirtinessIsCodeScoped:
    """A documentation edit cannot change a number, so it must not mark results unreproducible."""

    def test_only_code_paths_count(self, tmp_path, monkeypatch):
        import recognize.provenance as prov
        calls = []

        def fake_git(args):
            calls.append(args)
            return "" if args[0] == "status" else "abc1234"

        monkeypatch.setattr(prov, "_run_git", fake_git)
        assert prov._git_dirty() is False
        status = next(a for a in calls if a[0] == "status")
        assert status[-3:] == list(prov.CODE_PATHS) and "--" in status
        assert "docs" not in status

    def test_dirty_code_is_reported_with_the_files(self, monkeypatch):
        import recognize.provenance as prov
        monkeypatch.setattr(prov, "_run_git", lambda args: " M src/recognize/features.py" if args[0] == "status" else "abc1234")
        stamp = prov.stamp({}, None)
        assert stamp["git_dirty"] is True and "features.py" in stamp["git_dirty_files"]
