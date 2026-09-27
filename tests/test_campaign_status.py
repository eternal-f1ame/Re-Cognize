"""Campaign completion gate (scripts/report/campaign_status.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "report"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _run_dir(tmp_path, name="transreid_memory_seed0", epochs=200, dev_evals=20, nan=False,
             provenance=True, rising_loss=False, best_is_best=True):
    import campaign_status as cs
    from train import get_run
    run = get_run(name)
    d = tmp_path / run["backbone"] / run["config"] / f"seed{run['seed']}"
    d.mkdir(parents=True)
    losses = [3.0 - 0.01 * i for i in range(epochs)]
    if rising_loss:
        losses = [1.0 + 0.01 * i for i in range(epochs)]
    if nan:
        losses[10] = float("nan")
    devs = [{"epoch": (i + 1) * 10, "metric": "p2r1_map", "value": 0.4 + 0.001 * i} for i in range(dev_evals)]
    (d / "history.json").write_text(json.dumps({"train": [{"avg_loss": l} for l in losses], "dev": devs}))
    best_value = max(e["value"] for e in devs) if best_is_best else 0.0
    prov = {"git_commit": "abc1234", "pythonhashseed": "0",
            "dataset_counts": {"train_crops": 7668}} if provenance else {}
    torch.save({"provenance": prov, "dev_metric": {"metric": "p2r1_map", "value": best_value}}, d / "best.pth")
    return cs, run


def _point_at(monkeypatch, cs, tmp_path):
    """Send the gate, and the reported-checkpoint lookup it calls (train.output_dir), to the fixture's tree."""
    import train
    where = lambda r, mode="": tmp_path / r["backbone"] / r["config"] / f"seed{r['seed']}"
    monkeypatch.setattr(cs, "output_dir", where)
    monkeypatch.setattr(train, "output_dir", where)


class TestGate:
    def test_complete_run(self, tmp_path, monkeypatch):
        cs, run = _run_dir(tmp_path)
        _point_at(monkeypatch, cs, tmp_path)
        out = cs.check(run)
        assert out["state"] == "complete" and out["problems"] == [] and out["epochs"] == 200

    @pytest.mark.parametrize("kwargs,needle", [
        ({"nan": True}, "NaN"),
        ({"dev_evals": 5}, "dev evaluations"),
        ({"provenance": False}, "provenance"),
        ({"best_is_best": False}, "best dev epoch"),
    ])
    def test_defects_are_caught(self, tmp_path, monkeypatch, kwargs, needle):
        cs, run = _run_dir(tmp_path, **kwargs)
        _point_at(monkeypatch, cs, tmp_path)
        out = cs.check(run)
        assert out["state"] == "failed" and any(needle in p for p in out["problems"]), out

    def test_rising_loss_is_a_warning_not_a_failure(self, tmp_path, monkeypatch):
        """An ablation that removes a component may legitimately train worse (the no-WM run does)."""
        cs, run = _run_dir(tmp_path, rising_loss=True)
        _point_at(monkeypatch, cs, tmp_path)
        out = cs.check(run)
        assert out["state"] == "complete" and out["problems"] == []
        assert any("above the end of warmup" in w for w in out["warnings"])

    def test_expected_dev_count_matches_the_schedule(self, tmp_path, monkeypatch):
        """200 epochs evaluating every 10 gives 20 evaluations; the last epoch is already one of them."""
        cs, run = _run_dir(tmp_path, dev_evals=20)
        _point_at(monkeypatch, cs, tmp_path)
        assert cs.check(run)["state"] == "complete"

    def test_partial_run_is_running_not_failed(self, tmp_path, monkeypatch):
        cs, run = _run_dir(tmp_path, epochs=60, dev_evals=6)
        _point_at(monkeypatch, cs, tmp_path)
        assert cs.check(run)["state"] == "running"

    def test_missing_run_is_not_started(self, tmp_path, monkeypatch):
        import campaign_status as cs
        from train import get_run
        monkeypatch.setattr(cs, "output_dir", lambda r, mode="": tmp_path / "nothing" / r["name"])
        assert cs.check(get_run("magiv2_memory_seed0"))["state"] == "not started"

    def test_resume_lists_unfinished(self, tmp_path, monkeypatch, capsys):
        import campaign_status as cs
        monkeypatch.setattr(cs, "output_dir", lambda r, mode="": tmp_path / "nothing" / r["name"])
        assert cs.main(["--resume"]) == 0
        names = capsys.readouterr().out.split()
        from train import load_runs
        assert len(names) == len(load_runs()) and "transreid_memory_seed0" in names
