#!/usr/bin/env python3
"""Turn the one-epoch timing runs into results/timing.json and a wall-time budget.

    python scripts/train.py sbatch --timing            # produce the runs
    python scripts/report/timing.py --gpus 7           # summarise and budget the campaign
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from recognize.recipe import RECIPE  # noqa: E402
from train import load_runs  # noqa: E402  (scripts/train.py)


def read_timings(root: Path) -> Dict[str, Dict]:
    """backbone/config -> {seconds_per_epoch, dev_seconds, peak_gpu_mem_gb} from history.json."""
    out: Dict[str, Dict] = {}
    for hist_path in sorted(root.glob("*/*/seed*/history.json")):
        hist = json.loads(hist_path.read_text())
        if not hist.get("train"):
            continue
        rel = hist_path.relative_to(root)
        key = f"{rel.parts[0]}/{rel.parts[1]}"
        epoch = hist["train"][-1]
        out[key] = {
            "seconds_per_epoch": float(epoch.get("epoch_time_s") or epoch.get("train_time_s") or 0.0),
            "train_seconds": float(epoch.get("train_time_s") or 0.0),
            "dev_seconds": float(hist["dev"][-1].get("seconds", 0.0)) if hist.get("dev") else 0.0,
            "backbone": rel.parts[0], "config": rel.parts[1],
        }
    return out


def budget(timings: Dict[str, Dict], runs: List[Dict], gpus: int, epochs: int = RECIPE.epochs) -> Dict:
    """Serial GPU-hours for the campaign and the wall time on `gpus` GPUs (longest-first packing)."""
    per_run = []
    missing = []
    for r in runs:
        key = f"{r['backbone']}/{r['config']}"
        t = timings.get(key) or timings.get(f"{r['backbone']}/memory") or timings.get(f"{r['backbone']}/finetuned")
        if t is None:
            missing.append(r["name"]); continue
        # In a timing run the dev evaluation happens every epoch, so seconds_per_epoch already
        # contains one; the campaign evaluates every dev_eval_every epochs plus the last one.
        n_dev = epochs // RECIPE.dev_eval_every + 1
        hours = (t["train_seconds"] * epochs + t["dev_seconds"] * n_dev) / 3600.0
        per_run.append((r["name"], hours))
    lanes = [0.0] * max(1, gpus)
    for _, hours in sorted(per_run, key=lambda x: -x[1]):
        i = lanes.index(min(lanes)); lanes[i] += hours
    return {
        "gpu_hours_total": round(sum(h for _, h in per_run), 1),
        "wall_hours_on_gpus": round(max(lanes), 1) if per_run else 0.0,
        "gpus": gpus, "epochs": epochs, "n_runs": len(per_run),
        "per_run_hours": {n: round(h, 2) for n, h in sorted(per_run, key=lambda x: -x[1])},
        "missing_timings": missing,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing-root", type=Path, default=ROOT / "checkpoints" / "timing")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "timing.json")
    ap.add_argument("--gpus", type=int, default=7)
    args = ap.parse_args(argv)
    timings = read_timings(args.timing_root)
    if not timings:
        print(f"[timing] no history.json under {args.timing_root}; run `python scripts/train.py sbatch --timing` first")
        return 1
    report = {"timings": timings, "budget": budget(timings, load_runs(), args.gpus)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for k, v in sorted(timings.items()):
        print(f"  {k:<28} {v['seconds_per_epoch']:>7.1f} s/epoch  (train {v['train_seconds']:.1f}s, dev {v['dev_seconds']:.1f}s)")
    b = report["budget"]
    print(f"[timing] {b['n_runs']} runs x {b['epochs']} epochs = {b['gpu_hours_total']} GPU-hours "
          f"-> {b['wall_hours_on_gpus']} h on {b['gpus']} GPUs -> {args.out}")
    if b["missing_timings"]:
        print(f"[timing] no timing for: {', '.join(b['missing_timings'][:5])}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
