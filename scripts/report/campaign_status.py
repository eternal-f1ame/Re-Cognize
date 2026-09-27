#!/usr/bin/env python3
"""Status of the training campaign: one line per manifest run.

    python scripts/report/campaign_status.py            # complete / running / failed / not started
    python scripts/report/campaign_status.py --resume   # print the --runs list of everything unfinished
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from recognize.recipe import RECIPE  # noqa: E402
from train import load_runs, output_dir, reported_checkpoint  # noqa: E402  (scripts/train.py)


def check(run: Dict, mode: str = "", epochs: int = RECIPE.epochs) -> Dict:
    """Completion gate of one run.

    A run is complete when it has trained every epoch, logged every scheduled dev evaluation, and
    has its reported checkpoint and a best.pth whose provenance is complete (commit, pinned hash
    seed, dataset counts) and whose dev metric is the best in history.json.
    A NaN loss or a missing piece fails it; a final loss above the end of warmup is only a warning.
    """
    d = output_dir(run, mode)
    out = {"name": run["name"], "dir": d, "state": "not started", "epochs": 0, "dev": None, "problems": [], "warnings": []}
    hist_p, best_p = d / "history.json", d / "best.pth"
    reported = reported_checkpoint(run, mode)
    if not d.exists():
        return out
    epoch_ckpts = sorted(d.glob("epoch_*.pth"))
    if not hist_p.exists():
        out["state"] = "running" if epoch_ckpts else "not started"
        return out
    hist = json.loads(hist_p.read_text())
    losses = [e.get("avg_loss") for e in hist.get("train", [])]
    devs = hist.get("dev", [])
    out["epochs"] = len(losses)
    out["dev"] = round(max((e["value"] for e in devs), default=float("nan")), 4)
    # dev runs on every dev_eval_every-th epoch and on the last one; when the last epoch is
    # already a multiple, those coincide (200 epochs / every 10 -> 20 evaluations, not 21).
    expected_dev = len({e for e in range(1, epochs + 1) if e % RECIPE.dev_eval_every == 0 or e == epochs})
    if any(l is None or math.isnan(l) for l in losses):
        out["problems"].append("NaN loss")
    if len(losses) >= RECIPE.warmup_epochs + 1 and losses[-1] >= losses[RECIPE.warmup_epochs - 1]:
        # A warning, not a defect: an ablation that removes a component can legitimately train worse
        # (the no-working-memory run does exactly that). Only NaN and missing artefacts fail a run.
        out["warnings"].append(f"loss above the end of warmup ({losses[-1]:.3f} >= {losses[RECIPE.warmup_epochs - 1]:.3f})")
    if not reported.exists():
        out["problems"].append(f"no reported checkpoint ({reported.name})")
    if not best_p.exists():
        out["problems"].append("no best.pth")
    else:
        ck = torch.load(best_p, map_location="cpu", weights_only=False)
        prov = ck.get("provenance", {})
        if not prov.get("git_commit") or prov.get("pythonhashseed") != "0":
            out["problems"].append("provenance incomplete")
        if not prov.get("dataset_counts", {}).get("train_crops"):
            out["problems"].append("no dataset counts")
        if devs and ck.get("dev_metric") and abs(ck["dev_metric"]["value"] - max(e["value"] for e in devs)) > 1e-9:
            out["problems"].append("best.pth is not the best dev epoch")
    if len(losses) < epochs:
        out["state"] = "running"
    elif len(devs) < expected_dev:
        out["state"] = "failed"; out["problems"].append(f"{len(devs)}/{expected_dev} dev evaluations")
    else:
        out["state"] = "failed" if out["problems"] else "complete"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="", choices=("", "smoke", "timing"))
    ap.add_argument("--epochs", type=int, default=RECIPE.epochs)
    ap.add_argument("--resume", action="store_true", help="print the unfinished run names and nothing else")
    args = ap.parse_args(argv)
    rows: List[Dict] = [check(r, args.mode, args.epochs) for r in load_runs()]
    if args.resume:
        unfinished = [r["name"] for r in rows if r["state"] != "complete"]
        print(" ".join(unfinished))
        return 0
    by_state: Dict[str, int] = {}
    for r in rows:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
        flag = {"complete": "OK ", "running": "..", "failed": "!! ", "not started": "  "}[r["state"]]
        dev = "" if r["dev"] is None or math.isnan(r["dev"]) else f"dev {r['dev']:.4f}"
        notes = "; ".join(r["problems"] + [f"note: {w}" for w in r["warnings"]])
        print(f"{flag} {r['name']:<38} {r['state']:<12} {r['epochs']:>3}/{args.epochs} ep  {dev}"
              + (f"  <- {notes}" if notes else ""))
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(by_state.items())))
    return 0 if by_state.get("complete", 0) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
