#!/usr/bin/env python3
"""Enumerate the evaluation jobs: one array task per (model, corpus).

Models are the checkpoints of configs/runs.yaml plus the pretrained backbones with and
without memory. `--groups` selects the jobs: `popcharacters` is the full POPCharacters grid for
every model, `manga109` the zero-shot Manga109 evaluation of the grid runs, `reverse` P1 on
Re:Verse for the TransReID and MagiV2 grid runs, and `perturbations` the crop perturbations for
their finetuned and memory runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from recognize.backbones import BACKBONE_REGISTRY  # noqa: E402
from evaluate import split_for  # noqa: E402
from train import load_runs, reported_checkpoint  # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
FIELDS = ("group", "tag", "kind", "arg", "backbone", "batch", "corpus", "extra")
# The popcharacters group runs the full grid (every protocol, five gallery seeds, k = 1..5, the
# B_max sweep, the predicted/oracle/frozen policies, all five P3 rules); manga109 and reverse use
# the smaller grids.
FULL_GRID = ("--protocols p1 p2 p3 p4 --seeds 0 1 2 3 4 --k 1 2 3 4 5 --b-max-sweep "
             "--update-policies predicted oracle frozen "
             "--p3-rules fixed variance-adaptive density-aware cohesion-relative graph-louvain")
M109_GRID = "--protocols p1 p2 p3 p4 --seeds 0 1 2 --k 1 3 5 --p3-rules fixed"
P1_ONLY = "--protocols p1 --seeds 0 1 2 3 4"
GROUPS = ("popcharacters", "manga109", "reverse", "perturbations")


def build(groups: List[str], manifest: Optional[Path] = None) -> List[Dict]:
    rows: List[Dict] = []

    def add(group, tag, kind, arg, backbone, corpus, extra):
        rows.append({"group": group, "tag": tag, "kind": kind, "arg": str(arg), "backbone": backbone,
                     "batch": BATCH[backbone], "corpus": corpus, "extra": extra})

    runs = load_runs(manifest) if manifest else load_runs()

    if "popcharacters" in groups and manifest:
        # another run list (configs/runs_seeds.yaml): its runs on the headline grid, without the
        # pretrained rows (those belong to the campaign and do not depend on which checkpoints exist)
        for r in runs:
            add("popcharacters", r["name"], "checkpoint", reported_checkpoint(r), r["backbone"],
                "popcharacters", FULL_GRID)
        return rows

    if "popcharacters" in groups:
        for bb in sorted(BACKBONE_REGISTRY):
            add("popcharacters", f"pretrained__{bb}", "pretrained", bb, bb, "popcharacters", FULL_GRID + " --no-memory")
            add("popcharacters", f"pretrained__{bb}__memory", "pretrained", bb, bb, "popcharacters", FULL_GRID + " --memory")
        for r in runs:
            ckpt = reported_checkpoint(r)
            add("popcharacters", r["name"], "checkpoint", ckpt, r["backbone"], "popcharacters", FULL_GRID)
    if "manga109" in groups:
        for r in runs:
            if r["group"] == "grid":
                add("manga109", f"{r['name']}__manga109", "checkpoint", reported_checkpoint(r),
                    r["backbone"], "manga109", M109_GRID)
    if "reverse" in groups:
        for r in runs:
            if r["group"] == "grid" and r["backbone"] in ("magiv2", "transreid"):
                add("reverse", f"{r['name']}__reverse", "checkpoint", reported_checkpoint(r),
                    r["backbone"], "reverse", P1_ONLY)
    if "perturbations" in groups:
        from recognize.perturb import BOX_KINDS, PIXEL_KINDS
        for r in runs:
            if r["group"] == "grid" and r["backbone"] in ("magiv2", "transreid") and r["config"] in ("finetuned", "memory"):
                for kind in BOX_KINDS:
                    add("perturbations", f"{r['name']}__{kind}", "checkpoint", reported_checkpoint(r), r["backbone"],
                        "popcharacters", f"--protocols p1 p2 --seeds 0 1 2 --k 1 --box-noise {kind}")
                for kind in PIXEL_KINDS:
                    add("perturbations", f"{r['name']}__{kind}", "checkpoint", reported_checkpoint(r), r["backbone"],
                        "popcharacters", f"--protocols p1 p2 --seeds 0 1 2 --k 1 --pixel-noise {kind}")
    return rows


def drop_untrained(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split off rows whose checkpoint does not exist yet, so a half-trained campaign is submittable.

    A row is evaluable when its checkpoint is a finished-training artifact: `final.pth`, or the last
    epoch file of a run without one. `reported_checkpoint` falls back to `best.pth` only when neither
    exists, which means the run is still training: best.pth appears at the first dev evaluation, at
    epoch 10 of 200, so its presence is not evidence of a finished run.

    Never silent: main() prints every dropped row, and the caller decides. A `pretrained` row has no
    checkpoint to look for and always stays.
    """
    keep, drop = [], []
    for r in rows:
        done = r["kind"] != "checkpoint" or (Path(r["arg"]).exists() and Path(r["arg"]).name != "best.pth")
        (keep if done else drop).append(r)
    return keep, drop


def drop_done(rows: List[Dict], results_root: Path) -> Tuple[List[Dict], List[Dict]]:
    """Split off rows whose output tag already holds one JSON per series of that corpus.

    `evaluate.py --reuse-cached` skips a finished series anyway, but the task still starts, builds
    the model and exits, which costs a queue slot. This drops the row before submission. It counts
    files rather than validating them; a partially written tag keeps its row and --reuse-cached
    fills in the gaps.
    """
    expected = {}
    keep, done = [], []
    for r in rows:
        corpus = r["corpus"]
        if corpus not in expected:
            expected[corpus] = len(split_for(corpus, "test"))
        n = len(list((results_root / corpus / r["tag"]).glob("*.json")))
        (done if n >= expected[corpus] > 0 else keep).append(r)
    return keep, done


def write(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("\t".join(str(r[f]) for f in FIELDS) + "\n" for r in rows))


def read(path: Path) -> List[Dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            vals = line.split("\t")
            row = dict(zip(FIELDS, vals)); row["batch"] = int(row["batch"]); out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", nargs="+", default=["popcharacters"], choices=GROUPS)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="evaluate the runs of another run list (configs/runs_seeds.yaml) instead "
                         "of the campaign; only with --groups popcharacters, and no pretrained rows")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "eval_manifest.tsv")
    ap.add_argument("--require-checkpoint", action="store_true",
                    help="drop rows whose checkpoint file does not exist yet, and say which")
    ap.add_argument("--missing-only", action="store_true",
                    help="drop rows whose tag already has one result JSON per series")
    ap.add_argument("--results", type=Path, default=ROOT / "results",
                    help="results tree --missing-only checks against")
    args = ap.parse_args(argv)
    rows = build(args.groups, args.manifest)
    if args.require_checkpoint:
        rows, dropped = drop_untrained(rows)
        for r in dropped:
            print(f"     skipped (not trained yet)  {r['tag']:<46}{r['arg']}")
        if dropped:
            print(f"{len(dropped)} rows skipped: their checkpoint does not exist yet")
    if args.missing_only:
        rows, done = drop_done(rows, args.results)
        if done:
            print(f"{len(done)} rows skipped: already evaluated on every series")
    write(rows, args.out)
    for i, r in enumerate(rows):
        print(f"{i:3d}  {r['group']:<15}{r['tag']:<46}{r['corpus']:<15}batch {r['batch']}")
    print(f"{len(rows)} evaluation jobs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
