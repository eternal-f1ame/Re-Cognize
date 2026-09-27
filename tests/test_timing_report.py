"""Timing report and campaign budget."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "report"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _timing_tree(tmp_path):
    for bb, cfg, secs, dev in (("transreid", "memory", 60.0, 30.0), ("magiv2", "memory", 120.0, 45.0),
                               ("magiv3", "memory", 600.0, 90.0), ("reid5o", "memory", 90.0, 40.0),
                               ("instructreid", "memory", 60.0, 30.0)):
        d = tmp_path / bb / cfg / "seed0"; d.mkdir(parents=True)
        (d / "history.json").write_text(json.dumps({
            "train": [{"avg_loss": 1.0, "epoch_time_s": secs, "train_time_s": secs - 5}],
            "dev": [{"epoch": 1, "seconds": dev, "metric": "p2r1_map", "value": 0.4}]}))
    return tmp_path


class TestTiming:
    def test_reads_every_backbone(self, tmp_path):
        import timing
        t = timing.read_timings(_timing_tree(tmp_path))
        assert set(t) == {f"{b}/memory" for b in ("transreid", "magiv2", "magiv3", "reid5o", "instructreid")}
        assert t["magiv3/memory"]["seconds_per_epoch"] == 600.0 and t["magiv3/memory"]["dev_seconds"] == 90.0

    def test_budget_packs_the_campaign(self, tmp_path):
        import timing
        from train import load_runs
        t = timing.read_timings(_timing_tree(tmp_path))
        b = timing.budget(t, load_runs(), gpus=7, epochs=200)
        assert b["n_runs"] == len(load_runs()) and not b["missing_timings"]
        # magiv3 is the longest lane: 200 train epochs x 595 s + 21 dev evals x 90 s
        assert b["per_run_hours"]["magiv3_memory_seed0"] == pytest.approx((200 * 595 + 21 * 90) / 3600, abs=0.01)
        assert b["wall_hours_on_gpus"] >= max(b["per_run_hours"].values())
        assert b["gpu_hours_total"] > b["wall_hours_on_gpus"]

    def test_missing_timing_is_reported(self, tmp_path):
        import timing
        from train import load_runs
        partial = {k: v for k, v in timing.read_timings(_timing_tree(tmp_path)).items() if not k.startswith("magiv3")}
        b = timing.budget(partial, load_runs(), gpus=7)
        assert any("magiv3" in n for n in b["missing_timings"])
