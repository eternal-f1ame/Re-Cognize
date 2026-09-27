"""Recipe, schedule and checkpoint provenance."""
from __future__ import annotations

import sys

import pytest
import torch


def _parse(argv):
    from memory_block.training.train import parse_args
    saved = sys.argv; sys.argv = ["train.py", "--data-dir", "/tmp", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = saved


class TestSchedule:
    def test_warmup_then_cosine_to_zero_without_restarts(self):
        from memory_block.training.train import lr_multiplier
        from recognize.recipe import RECIPE
        curve = [lr_multiplier(e, RECIPE.warmup_epochs, RECIPE.epochs) for e in range(RECIPE.epochs)]
        assert curve[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
        assert all(b < a for a, b in zip(curve[4:], curve[5:]))           # strictly decreasing after the peak
        assert curve[-1] == pytest.approx(0.0, abs=1e-12) and max(curve) == 1.0 and curve.count(1.0) == 1

    def test_lambda_lr_matches_the_curve(self):
        from memory_block.training.train import get_warmup_cosine_scheduler, lr_multiplier
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        sch = get_warmup_cosine_scheduler(opt, 5, 200)
        seen = []
        for e in range(200):
            seen.append(opt.param_groups[0]["lr"]); sch.step()
        assert seen == pytest.approx([1e-4 * lr_multiplier(e, 5, 200) for e in range(200)])


class TestDefaults:
    def test_cli_defaults_equal_the_recipe(self):
        from recognize.recipe import CLI_FIELDS, RECIPE
        args = vars(_parse([]))
        for dest, field in CLI_FIELDS.items():
            assert args[dest] == getattr(RECIPE, field), (dest, args[dest], getattr(RECIPE, field))
        assert "restart_period" not in args and "lora_lr_scale" not in args
        assert _parse(["--no-amp"]).amp is False and _parse(["--dev-eval-every", "3"]).val_freq == 3

    def test_launcher_emits_the_recipe(self):
        import importlib
        from recognize.recipe import RECIPE
        launcher = importlib.import_module("train")           # scripts/train.py
        cmd = launcher.build_command(launcher.get_run("transreid_memory_seed0"))
        flag = lambda f: cmd[cmd.index(f) + 1]
        assert flag("--epochs") == str(RECIPE.epochs) and flag("--lr") == str(RECIPE.lr)
        assert flag("--dev-eval-every") == str(RECIPE.dev_eval_every) and flag("--save-freq") == str(RECIPE.save_every)
        assert flag("--warmup-epochs") == str(RECIPE.warmup_epochs) and flag("--weight-decay") == str(RECIPE.weight_decay)


class TestPayload:
    def test_checkpoint_payload_carries_provenance(self, monkeypatch):
        import argparse
        from memory_block.training.train import checkpoint_payload, get_warmup_cosine_scheduler
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        m = torch.nn.Linear(2, 2); opt = torch.optim.AdamW(m.parameters(), lr=1e-4); sch = get_warmup_cosine_scheduler(opt, 5, 200)
        args = argparse.Namespace(epochs=200, lr=1e-4, seed=0, name="x")
        pl = checkpoint_payload(m, opt, sch, None, 7, {"c": 1}, args, {"train_crops": 10}, {"metric": "p1_map", "value": 0.5}, "memory")
        assert pl["epoch"] == 7 and pl["args"]["epochs"] == 200 and pl["mAP"] == 0.5
        prov = pl["provenance"]
        for key in ("git_commit", "git_dirty", "pythonhashseed", "args", "dataset_counts", "dev_metric"):
            assert key in prov
        assert prov["dataset_counts"] == {"train_crops": 10} and prov["pythonhashseed"] == "0"

    def test_training_refuses_unpinned_hashseed(self, monkeypatch, tmp_path):
        from memory_block.training.train import train_memory_model
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
            train_memory_model(_parse(["--split-config", str(tmp_path / "none.yaml")]))
