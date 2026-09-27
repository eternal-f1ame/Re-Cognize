"""End-to-end training loop on synthetic data with a tiny custom backbone: dev-only selection."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch
import yaml
from PIL import Image


def _series(root, name, identities, crops_per_identity, seed):
    d = root / name; (d / "images").mkdir(parents=True); (d / "annotations").mkdir()
    (d / "category_mapping.json").write_text(json.dumps({str(i): f"{name}-{i}" for i in identities}))
    rng = np.random.default_rng(seed)
    for page in range(crops_per_identity):
        stem = f"{name} - c001 - p{page + 1:03d}"
        Image.fromarray((rng.random((64, 96, 3)) * 255).astype(np.uint8)).save(d / "images" / f"{stem}.jpg")
        rows = [f"{i} {0.15 + 0.25 * j:.2f} 0.5 0.2 0.6" for j, i in enumerate(identities)]
        (d / "annotations" / f"{stem}.txt").write_text("\n".join(rows) + "\n")


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "Datasets"
    _series(root, "Train A", [0, 1, 2], 6, 1)
    _series(root, "Train B", [0, 1, 2], 6, 2)
    _series(root, "Dev A", [0, 1], 4, 3)
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"train": ["Train A", "Train B"], "dev": ["Dev A"], "test": ["Test Never Read"]}))
    return root, split


def _args(root, split, out, extra=()):
    from memory_block.training.train import parse_args
    argv = ["--data-dir", str(root), "--split-config", str(split), "--backbone", "custom",
            "--backbone-module", "_tiny_backbone", "--backbone-class", "TinyBackbone", "--backbone-kwargs", '{"feat_dim": 8}',
            "--feat-dim", "8", "--height", "16", "--width", "16", "--epochs", "2", "--val-freq", "1", "--save-freq", "1",
            "--pk-sampling", "--p", "2", "--k", "4", "--k-support", "2", "--ce-weight", "0", "--no-wandb", "--workers", "0",
            "--device", "cpu", "--name", "run", "--output-dir", str(out), "--seed", "0", *extra]
    saved = sys.argv
    sys.argv = ["train.py"] + argv
    try:
        return parse_args()
    finally:
        sys.argv = saved


class TestTrainingLoop:
    def test_memory_run_selects_on_dev_only(self, data, tmp_path, monkeypatch):
        import memory_block.training.train as T
        root, split = data
        seen = []
        real = T.dev_score

        def spy(model, transform, series_dirs, **kw):
            seen.append([p.name for p in series_dirs]); return real(model, transform, series_dirs, **kw)

        monkeypatch.setattr(T, "dev_score", spy)
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        T.train_memory_model(_args(root, split, tmp_path / "out"))
        assert seen == [["Dev A"], ["Dev A"]]                                   # two dev evals, dev series only
        out = tmp_path / "out"
        hist = json.loads((out / "history.json").read_text())
        assert [h["epoch"] for h in hist["dev"]] == [1, 2] and hist["dev"][0]["metric"] == "p2r1_map"
        ck = torch.load(out / "best.pth", map_location="cpu", weights_only=False)
        assert ck["dev_metric"]["metric"] == "p2r1_map" and ck["training_mode"] == "memory"
        assert not (root / "Test Never Read").exists()

    def test_baseline_run_uses_p1(self, data, tmp_path, monkeypatch):
        import memory_block.training.train as T
        root, split = data
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        T.train_memory_model(_args(root, split, tmp_path / "out", extra=("--no-memory",)))
        hist = json.loads((tmp_path / "out" / "history.json").read_text())
        assert hist["dev"][-1]["metric"] == "p1_map"

    def test_split_without_dev_is_rejected(self, data, tmp_path):
        from memory_block.training.train import load_training_split
        bad = tmp_path / "bad.yaml"; bad.write_text(yaml.safe_dump({"train": ["Train A"], "val": ["Dev A"]}))
        with pytest.raises(ValueError, match="dev"):
            load_training_split(bad)
        overlap = tmp_path / "overlap.yaml"; overlap.write_text(yaml.safe_dump({"train": ["A"], "dev": ["A"], "test": ["B"]}))
        with pytest.raises(ValueError):
            load_training_split(overlap)


class TestPerStepDeterminism:
    """A step's randomness must depend on (seed, epoch, batch), not on what the RNG did before."""

    def test_step_seed_is_a_pure_function_of_position(self):
        from memory_block.training.train import step_seed
        assert step_seed(0, 0, 0) != step_seed(0, 0, 1) != step_seed(0, 1, 0)
        assert step_seed(1, 3, 7) == step_seed(1, 3, 7)
        assert step_seed(0, 3, 7) != step_seed(1, 3, 7)
        assert all(0 <= step_seed(s, e, b) < 2 ** 31 - 1 for s in (0, 1, 2) for e in (0, 199) for b in (0, 18))

    def test_training_is_unaffected_by_prior_rng_state(self, data, tmp_path, monkeypatch):
        """The whole point: consuming the global RNG before training must not change the result."""
        import memory_block.training.train as T
        root, split = data
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        outs = []
        for extra_draws in (0, 137):
            torch.manual_seed(12345)
            for _ in range(extra_draws):
                torch.rand(1)
            out = tmp_path / f"run{extra_draws}"
            T.train_memory_model(_args(root, split, out))
            ck = torch.load(out / "final.pth", map_location="cpu", weights_only=False)
            outs.append(ck["model_state_dict"]["memory_block.fusion.gate.0.weight"].clone())
        torch.testing.assert_close(outs[0], outs[1])

    def test_final_checkpoint_is_the_last_epoch(self, data, tmp_path, monkeypatch):
        import memory_block.training.train as T
        root, split = data
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        out = tmp_path / "run"
        T.train_memory_model(_args(root, split, out))
        final = torch.load(out / "final.pth", map_location="cpu", weights_only=False)
        best = torch.load(out / "best.pth", map_location="cpu", weights_only=False)
        last = torch.load(out / "epoch_0002.pth", map_location="cpu", weights_only=False)
        assert final["epoch"] == 2 and final["provenance"]["pythonhashseed"] == "0"
        for k, v in final["model_state_dict"].items():
            torch.testing.assert_close(v, last["model_state_dict"][k])
        assert best["epoch"] in (1, 2)          # best.pth still exists, still dev-selected
