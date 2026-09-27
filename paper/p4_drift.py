"""P4 over the stream: per-quartile identity Rank-1 under the three update policies, and contamination.

The appendix section on P4 update dynamics states these as numbers; this figure draws them as curves.
Both come from the same fields of the P4 results, `r1_by_quartile` and `contamination`, at the
protocol's operating point (k = 1, B_max = 50), macro over the 8 test series, mean over five seed
draws and three training runs, for the five backbones with the memory block:

  (a), (b)  identity Rank-1 in each quarter of the stream, pooled over the five backbones, for the
            static gallery (the frozen policy, i.e. P2), growth by the model's own top-1 (predicted)
            and growth by the true label (oracle), under random and chronological seeding;
  (c)       the share of the grown gallery that is mislabelled at the end of the stream, per backbone.

    python paper/p4_drift.py --out paper/generated/figures
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42    # TrueType: every glyph stays text
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from paper_style import TYPE, PAGE_W, BACKBONES, BACKBONE_LABELS, save_fixed  # noqa: E402
from _config import safe_name                                                 # noqa: E402
from recognize.data import load_split                                      # noqa: E402

RESULTS = ROOT / "results" / "popcharacters"
TRAIN_SEEDS, EVAL_SEEDS, K, B_MAX, CONFIG = (0, 1, 2), range(5), "1", "50", "memory"
POLICIES = [("frozen", "static", "#8c8c8c", "-"), ("predicted", "predicted", "#C44E52", "-"),
            ("oracle", "oracle", "#55A868", "-")]
SEEDING = [("random", "random seeding", "#4C72B0"), ("temporal", "chronological seeding", "#8172B3")]
GRID, INK = "#d9d9d9", "#1b1b1b"


def entries(bb: str, strategy: str, policy: str):
    """Every (training run, series, seed draw) P4 entry of one cell."""
    for t in TRAIN_SEEDS:
        for name in load_split()["test"]:
            d = json.loads((RESULTS / f"{bb}_{CONFIG}_seed{t}" / f"{safe_name(name)}.json").read_text())
            yield t, name, [d["p4"][strategy][K][str(s)][policy][B_MAX] for s in EVAL_SEEDS]


def cell(bb: str, strategy: str, policy: str, value) -> float:
    """Mean over training runs of the macro over series of the mean over seed draws."""
    per_run = {}
    for t, _, es in entries(bb, strategy, policy):
        per_run.setdefault(t, []).append(statistics.fmean(value(e) for e in es))
    return 100 * statistics.fmean(statistics.fmean(v) for v in per_run.values())


def style(ax):
    ax.tick_params(colors=INK, labelsize=TYPE["tick"], length=2.2, color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def figure(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.0), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    quarters = np.arange(1, 5)
    for ax, (strategy, title, _) in zip(axes[:2], SEEDING):
        for policy, label, colour, ls in POLICIES:
            ys = [statistics.fmean(cell(bb, strategy, policy, lambda e, q=q: e["r1_by_quartile"][q])
                                   for bb in BACKBONES) for q in range(4)]
            ax.plot(quarters, ys, color=colour, ls=ls, lw=1.1, marker="o", ms=2.6, label=label)
        ax.set_title(title, fontsize=TYPE["label"], pad=3)
        ax.set_xticks(quarters)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("quarter of the stream", fontsize=TYPE["label"])
        ax.set_ylim(0, 60)
        style(ax)
    axes[0].set_ylabel("identity Rank-1 (%)", fontsize=TYPE["label"])
    axes[1].tick_params(labelleft=False)

    ax = axes[2]
    x = np.arange(len(BACKBONES))
    w = 0.38
    for j, (strategy, title, colour) in enumerate(SEEDING):
        ys = [cell(bb, strategy, "predicted", lambda e: e["contamination"]) for bb in BACKBONES]
        ax.bar(x + (j - 0.5) * w, ys, width=w, color=colour, label=title.split()[0])
    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_LABELS[b] for b in BACKBONES], rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=TYPE["tick"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("mislabelled at end (%)", fontsize=TYPE["label"])
    ax.set_title("contamination, predicted", fontsize=TYPE["label"], pad=3)
    style(ax)

    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="upper center", ncol=5, frameon=False, fontsize=TYPE["legend"],
               handlelength=1.4, handletextpad=0.4, columnspacing=1.1, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9), w_pad=0.8)
    save_fixed(fig, out, pad=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "generated" / "figures")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    figure(a.out / "p4_drift.pdf")
    print("[fig] p4_drift.pdf  quartile Rank-1 x 3 policies x 2 seedings, contamination per backbone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
