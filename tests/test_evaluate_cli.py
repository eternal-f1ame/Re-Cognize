"""End-to-end test of scripts/evaluate.py on a synthetic series with a stub model."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from test_features import StubModel  # tests/ is on sys.path via conftest
from recognize.checkpoints import LoadedModel


def _lm(model, backbone="transreid"):
    return LoadedModel(model=model, backbone=backbone, checkpoint=None, normalize="imagenet", flags={"training_mode": "pretrained", "mask_ratio": None})


@pytest.fixture
def series_root(tmp_path):
    """Root holding one series 'X' with two pages, three boxes, two characters (Bob x2, Ann x1)."""
    d = tmp_path / "root" / "X"
    (d / "images").mkdir(parents=True); (d / "annotations").mkdir()
    (d / "category_mapping.json").write_text(json.dumps({"0": "Ann", "3": "Bob"}))
    rng = np.random.default_rng(0)
    for stem in ("X - c001 (web) - p010 [Unknown]", "X - c001 (web) - p002 [Unknown]"):
        Image.fromarray((rng.random((100, 200, 3)) * 255).astype(np.uint8)).save(d / "images" / f"{stem}.jpg")
    (d / "annotations" / "X - c001 (web) - p010 [Unknown].txt").write_text("3 0.5 0.5 0.2 0.4\n")
    (d / "annotations" / "X - c001 (web) - p002 [Unknown].txt").write_text("3 0.25 0.5 0.5 0.5\n0 0.9 0.5 0.1 0.2\n")
    return tmp_path / "root"


def _six_vector(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.tensor(np.concatenate([a.mean((0, 1)), a.std((0, 1))]), dtype=torch.float32)


class TestSchema:
    def test_rejects_missing_provenance_and_blocks(self):
        from recognize.schema import validate
        with pytest.raises(ValueError, match="provenance"):
            validate({"p1": {"0": {"mAP": 1.0}}})
        prov = {k: 1 for k in ("git_commit", "git_dirty", "seeds", "checkpoint_sha256", "n_crops", "n_identities")}
        prov.update({"pythonhashseed": "0", "args": {"protocols": ["p1", "p3"]}})
        good = {"provenance": prov, "series": {"name": "X", "n_crops": 3, "n_identities": 2},
                "p1": {"0": {"mAP": 1.0}}, "p3": {"fixed": {"clusters": 2}}}
        validate(good)
        bad = dict(good); bad.pop("p3")
        with pytest.raises(ValueError, match="p3"):
            validate(bad)

    def test_accepts_a_result_without_repository_state(self):
        # the published results carry no git commit or tree state
        from recognize.schema import validate
        prov = {k: 1 for k in ("seeds", "checkpoint_sha256", "n_crops", "n_identities")}
        prov.update({"pythonhashseed": "0", "args": {"protocols": ["p1"]}})
        validate({"provenance": prov, "series": {"name": "X", "n_crops": 3, "n_identities": 2},
                  "p1": {"0": {"mAP": 1.0}}})


class TestCLI:
    def test_end_to_end_memory_stub(self, series_root, tmp_path, monkeypatch):
        import evaluate
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        monkeypatch.setattr(evaluate, "build_model", lambda args, device: _lm(StubModel()))
        monkeypatch.setattr(evaluate, "build_transform", lambda backbone, normalize: _six_vector)
        out = tmp_path / "out"
        rc = evaluate.main(["--pretrained", "transreid", "--data-root", str(series_root), "--series", "X",
                            "--protocols", "p1", "p2", "p3", "p4", "--seeds", "0", "1", "--k", "1", "--b-max-sweep",
                            "--update-policies", "predicted", "oracle", "frozen", "--out", str(out)])
        assert rc == 0
        result = json.loads((out / "X.json").read_text())
        from recognize.schema import validate
        validate(result)
        assert result["provenance"]["pythonhashseed"] == "0"
        assert result["series"]["n_crops"] == 3 and result["series"]["n_identities"] == 2
        assert set(result["p1"]) == {"0", "1"} and result["p1"]["0"]["n_excluded_singletons"] == 1
        assert result["p2"]["temporal"]["1"]["0"]["n_gallery_only_identities"] == 1
        assert set(result["p4"]["random"]["1"]["0"]["oracle"]) == {"50", "0", "5", "10", "25", "100", "inf"}
        assert result["p3"]["fixed"]["clusters"] >= 1 and result["provenance"]["args"]["use_memory"] is True
        assert result["provenance"]["args"]["normalize"] == "imagenet" and result["provenance"]["args"]["model_flags"]["training_mode"] == "pretrained"

    def test_reuse_cached_skips_existing(self, series_root, tmp_path, monkeypatch, capsys):
        import evaluate
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        monkeypatch.setattr(evaluate, "build_model", lambda args, device: _lm(StubModel(with_memory=False)))
        monkeypatch.setattr(evaluate, "build_transform", lambda backbone, normalize: _six_vector)
        out = tmp_path / "out"
        base = ["--pretrained", "transreid", "--data-root", str(series_root), "--series", "X", "--protocols", "p1",
                "--seeds", "0", "--out", str(out)]
        evaluate.main(base)
        first = (out / "X.json").read_text()
        evaluate.main(base + ["--reuse-cached"])
        assert "reusing" in capsys.readouterr().out and (out / "X.json").read_text() == first
        evaluate.main(base)                                   # default recomputes (timestamps differ)
        assert json.loads((out / "X.json").read_text())["p1"] == json.loads(first)["p1"]

    def test_pretrained_uses_identity_bnneck(self, monkeypatch):
        import evaluate
        from recognize.features import _L2Neck
        monkeypatch.setattr(evaluate, "construct_model", lambda cfg: StubModel(with_memory=False))
        args = evaluate.build_parser().parse_args(["--pretrained", "transreid"])
        lm = evaluate.build_model(args, "cpu")
        assert isinstance(lm.model.bnneck, _L2Neck) and lm.backbone == "transreid" and lm.checkpoint is None
        assert lm.normalize == "imagenet" and lm.flags["training_mode"] == "pretrained" and lm.flags["config_label"] == "Pretrained"
        seen = {}
        monkeypatch.setattr(evaluate, "construct_model", lambda cfg: seen.setdefault("cfg", cfg) and StubModel(with_memory=False))
        lm5 = evaluate.build_model(evaluate.build_parser().parse_args(["--pretrained", "reid5o"]), "cpu")
        cfg = seen["cfg"]
        assert lm5.normalize == "clip" and cfg.feat_dim == 512 and cfg.backbone_type == "reid5o"
        # weights, config path, stride and revision are resolved inside the builders from the registry
        assert cfg.backbone_config is None and cfg.backbone_checkpoint is None

    def test_requires_pinned_hashseed(self, monkeypatch):
        import evaluate
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
            evaluate.main(["--pretrained", "transreid"])


class TestSeriesResolution:
    """Each corpus resolves its own held-out list; only POPCharacters has a dev split."""

    def test_popcharacters_uses_the_split_file(self):
        import evaluate
        from recognize.data import load_split
        assert evaluate.split_for("popcharacters", "test") == load_split()["test"]
        assert evaluate.split_for("popcharacters", "dev") == load_split()["dev"]

    def test_manga109_uses_the_registry_and_has_no_dev(self):
        import evaluate
        from _config import DATASET_REGISTRY
        series = evaluate.split_for("manga109", "test")
        assert series == list(DATASET_REGISTRY["manga109"].test_manga) and len(series) == 27
        with pytest.raises(SystemExit, match="no dev split"):
            evaluate.split_for("manga109", "dev")


class TestAtomicWrite:
    """A result file is either the previous version or the new one, never half of either."""

    def test_the_temporary_file_is_gone_and_the_content_lands(self, tmp_path):
        import evaluate
        p = tmp_path / "Bakuman.json"
        evaluate.write_atomic(p, '{"a": 1}')
        assert p.read_text() == '{"a": 1}'
        assert list(tmp_path.iterdir()) == [p]

    def test_it_replaces_an_existing_file(self, tmp_path):
        import evaluate
        p = tmp_path / "Bakuman.json"
        p.write_text("old")
        evaluate.write_atomic(p, "new")
        assert p.read_text() == "new" and list(tmp_path.iterdir()) == [p]

    def test_two_writers_do_not_share_a_temporary_name(self, tmp_path, monkeypatch):
        # the pid is in the temp name, so two processes writing one tag cannot truncate each other
        from pathlib import Path
        import evaluate
        seen = []
        real = Path.write_text

        def spy(self, text, *a, **k):
            seen.append(self.name)
            return real(self, text, *a, **k)

        monkeypatch.setattr(Path, "write_text", spy)
        monkeypatch.setattr(evaluate.os, "getpid", lambda: 111)
        evaluate.write_atomic(tmp_path / "x.json", "a")
        monkeypatch.setattr(evaluate.os, "getpid", lambda: 222)
        evaluate.write_atomic(tmp_path / "x.json", "b")
        assert seen == [".x.json.111.tmp", ".x.json.222.tmp"]
        assert (tmp_path / "x.json").read_text() == "b"
