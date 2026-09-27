#!/usr/bin/env python3
"""What crop perturbation costs each model, and whether memory buys anything under it.

One table per protocol. Rows are perturbations, columns are the four models; each cell is the
change from the same model's clean number, so the columns are comparable even though the models
are not. The last pair of columns is the question the memory block exists to answer: does its
advantage over the finetuned baseline grow when the crop degrades?

    python scripts/report/robustness.py [--out results/reports/robustness.md]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "report"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tables as T                                        # noqa: E402
from recognize.data import load_split                  # noqa: E402
from recognize.perturb import BOX_KINDS, PIXEL_KINDS      # noqa: E402

MODELS = ["transreid_finetuned_seed0", "transreid_memory_seed0",
          "magiv2_finetuned_seed0", "magiv2_memory_seed0"]
LABEL = {"transreid_finetuned_seed0": "TransReID FT", "transreid_memory_seed0": "TransReID FT+Mem",
         "magiv2_finetuned_seed0": "MagiV2 FT", "magiv2_memory_seed0": "MagiV2 FT+Mem"}
PAIRS = [("transreid", "transreid_finetuned_seed0", "transreid_memory_seed0"),
         ("magiv2", "magiv2_finetuned_seed0", "magiv2_memory_seed0")]


def value(root: Path, tag: str, series: List[str], protocol: str, **kw) -> Optional[float]:
    """Macro over series, mean over seeds, in points; None unless every series is in."""
    c = T.cell(T.load_series(root, tag, series), protocol, **kw)
    if c is None or c["n_series"] != len(series):
        return None
    return c["mean"] * 100


def table(root: Path, series: List[str], protocol: str, title: str, **kw) -> str:
    clean = {m: value(root, m, series, protocol, **kw) for m in MODELS}
    out = [f"### {title}", "",
           "Clean number first, then the change under each perturbation.", "",
           "| perturbation | " + " | ".join(LABEL[m] for m in MODELS) + " | "
           + " | ".join(f"mem-FT, {bb}" for bb, _, _ in PAIRS) + " |",
           "|---|" + "---|" * (len(MODELS) + len(PAIRS))]
    row = ["| clean |"]
    for m in MODELS:
        row.append(f" {clean[m]:.2f} |" if clean[m] is not None else " n/a |")
    for _, ft, mem in PAIRS:
        row.append(f" {clean[mem] - clean[ft]:+.2f} |" if None not in (clean[ft], clean[mem]) else " n/a |")
    out.append("".join(row))
    for kind in list(BOX_KINDS) + list(PIXEL_KINDS):
        vals = {m: value(root, f"{m}__{kind}", series, protocol, **kw) for m in MODELS}
        if all(v is None for v in vals.values()):
            continue
        row = [f"| {kind} |"]
        for m in MODELS:
            row.append(f" {vals[m] - clean[m]:+.2f} |" if None not in (vals[m], clean[m]) else " n/a |")
        for _, ft, mem in PAIRS:
            row.append(f" {vals[mem] - vals[ft]:+.2f} |" if None not in (vals[ft], vals[mem]) else " n/a |")
        out.append("".join(row))
    return "\n".join(out) + "\n\n"


def summary(root: Path, series: List[str]) -> str:
    """Does the memory's advantage grow as the crop degrades?"""
    lines = ["### Does the memory earn more under perturbation?", ""]
    for bb, ft, mem in PAIRS:
        clean = value(root, mem, series, "p1")
        base = value(root, ft, series, "p1")
        if None in (clean, base):
            continue
        gaps = []
        for kind in list(BOX_KINDS) + list(PIXEL_KINDS):
            a, b = value(root, f"{ft}__{kind}", series, "p1"), value(root, f"{mem}__{kind}", series, "p1")
            if None not in (a, b):
                gaps.append(b - a)
        if not gaps:
            continue
        lines.append(f"- **{bb}**: the memory is worth {clean - base:+.2f} mAP on clean crops and "
                     f"{statistics.fmean(gaps):+.2f} on average across {len(gaps)} perturbations "
                     f"(range {min(gaps):+.2f} to {max(gaps):+.2f}). "
                     + ("It earns more as the crop degrades."
                        if statistics.fmean(gaps) > (clean - base) else
                        "It does not earn more as the crop degrades."))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "popcharacters")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "reports" / "robustness.md")
    args = ap.parse_args(argv)
    series = load_split()["test"]
    text = ["# Crop robustness", "",
            "Perturbations are applied to the crop box or its pixels before the backbone sees it "
            "(`recognize.perturb`); every model is evaluated on the same perturbed stream, three "
            "gallery seeds, k=1 for P2. Values are points of mAP, macro over the "
            f"{len(series)} test series.", ""]
    text.append(table(args.results, series, "p1", "P1 closed set, mAP"))
    text.append(table(args.results, series, "p2", "P2 Seq-R at k=1, mAP", strategy="random", k="1"))
    text.append(summary(args.results, series))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(text))
    print(f"[robustness] wrote {args.out}")
    print("\n".join(text[-2:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
