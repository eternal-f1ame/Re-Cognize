"""Registry facts and the weight fetcher."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))


class TestRegistry:
    def test_values_match_the_protocol(self):
        from recognize.backbones import BACKBONE_REGISTRY as R
        assert set(R) == {"transreid", "magiv2", "magiv3", "instructreid", "reid5o"}
        assert {k: (c.native_dim, c.height, c.width, c.normalize) for k, c in R.items()} == {
            "transreid": (768, 256, 128, "imagenet"), "magiv2": (768, 224, 224, "imagenet"),
            "magiv3": (1024, 384, 384, "imagenet"), "instructreid": (768, 256, 128, "imagenet"),
            "reid5o": (512, 384, 128, "clip")}
        assert all(c.k == 4 for c in R.values()) and R["magiv3"].p == 4 and R["transreid"].p == 8
        assert R["transreid"].patch_stride == 12 and R["instructreid"].patch_stride == 16
        assert R["magiv2"].hf_revision and R["magiv3"].hf_revision
        for k in ("transreid", "instructreid", "reid5o"):
            assert R[k].weights and R[k].weights_sha256 and len(R[k].weights_sha256) == 64
        assert R["transreid"].weights_sources[0].startswith("gdrive:") and R["instructreid"].weights_sources[0].startswith("gdrive:")

    def test_aliases_and_reexport(self):
        from recognize.backbones import BACKBONE_REGISTRY as R
        import _config
        assert _config.BACKBONE_REGISTRY is R and _config.ALL_BACKBONE_NAMES == list(R)
        assert R["magiv3"].feat_dim == 1024 and R["transreid"].checkpoint == R["transreid"].weights
        assert R["reid5o"].weights_path() == REPO / R["reid5o"].weights


class TestFetch:
    @pytest.fixture
    def sandbox(self, tmp_path, monkeypatch):
        import fetch_weights as fw
        from recognize import backbones
        cfg = backbones.BackboneConfig(name="fake", height=8, width=8, native_dim=4, weights="w/fake.pth",
                                       weights_sources=("gdrive:bad", "https://example.org/fake.pth"),
                                       weights_sha256=hashlib.sha256(b"payload").hexdigest())
        monkeypatch.setitem(fw.BACKBONE_REGISTRY, "fake", cfg)
        monkeypatch.setattr(fw, "ROOT", tmp_path)
        monkeypatch.setattr(fw, "SUMS", tmp_path / "w" / "SHA256SUMS")
        monkeypatch.setattr(backbones, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(cfg.__class__, "weights_path", lambda self: tmp_path / self.weights)
        return fw, cfg

    def test_tries_sources_in_order_and_pins_sha(self, sandbox, tmp_path):
        fw, cfg = sandbox
        calls = []

        def downloader(src, dest):
            calls.append(src)
            if src.startswith("gdrive:"):
                raise RuntimeError("quota")
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(b"payload")

        out = fw.fetch("fake", downloader=downloader)
        assert out == tmp_path / "w" / "fake.pth" and calls == ["gdrive:bad", "https://example.org/fake.pth"]
        assert (tmp_path / "w" / "SHA256SUMS").read_text().strip().endswith("w/fake.pth")
        assert fw.fetch("fake", downloader=lambda s, d: pytest.fail("must not re-download")) == out

    def test_sha_mismatch_is_refused(self, sandbox, tmp_path):
        fw, cfg = sandbox

        def downloader(src, dest):
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(b"tampered")

        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            fw.fetch("fake", downloader=downloader)
