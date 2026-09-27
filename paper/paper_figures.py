"""Three of the four panels of the paper's Figure 3, the headroom figure, from `results/`.

  ceiling_recovery.pdf  (a) what one seed per identity (P2-R at k=1) recovers of the closed-set P1
                        ceiling, on mAP and on Rank-1. It *exceeds* the mAP ceiling (102 to 107 %)
                        while recovering only 41 to 66 % of closed-set Rank-1.
  p4_closes_gap.pdf     (b) P4 identity Rank-1 at k=1 for the static gallery, growth by top-1 and
                        growth by true label. Growth with predicted labels is below the static
                        gallery on all five backbones, and the oracle bar shows the size of the
                        acceptance gap.
  adaptation.pdf     (c) the P1 mAP the released backbone gains from BNNeck fine-tuning and then
                        from the memory block.

The fourth panel, one_breakeven.pdf, comes from recast_figures.py.

The three panels print at 0.48\\linewidth (paper_style.fig_size('half'), 2.64 x 1.32 in) beside a
fourth, so they share one plot box: the same margins, a two-row legend band above the axes, and the
five backbone names on two staggered rows, which is what lets them stay horizontal at 7 pt. Every
text span prints at paper_style.FLOOR or above (paper/figure_audit.py checks it).

    python paper/paper_figures.py --out paper/generated/figures
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42            # TrueType: every glyph stays text
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
import numpy as np                                  # noqa: E402

from recognize.data import load_split            # noqa: E402
from report import tables as T                      # noqa: E402
from paper_style import TYPE, fig_size, save_fixed, legend_above  # noqa: E402

BACKBONES = [("transreid", "TransReID"), ("magiv2", "MagiV2"), ("magiv3", "MagiV3"),
             ("instructreid", "InstructReID"), ("reid5o", "ReID5o")]
INK, GRID = "#1b1b1b", "#d9d9d9"

# One plot box for all three panels, in points from the canvas edges. Left holds a two-line axis
# label and three-digit ticks, top a two-row legend, bottom the two staggered rows of names.
W_PT, H_PT = (v * 72 for v in fig_size("half"))
LEFT, RIGHT, TOP, BOTTOM = 40.0, 5.0, 24.0, 25.0
TICK_PAD = 2.5


def val(root: Path, series: List[str], bb: str, cfg: str, proto: str,
        metric: str = "mAP", **kw) -> Optional[float]:
    v = T.by_train_seed(root, series, bb, cfg, proto, metric, seeds=(0, 1, 2), **kw)
    return statistics.fmean(v.values()) * 100 if v else None


def panel():
    fig = plt.figure(figsize=fig_size("half"))
    ax = fig.add_axes([LEFT / W_PT, BOTTOM / H_PT,
                       1 - (LEFT + RIGHT) / W_PT, 1 - (TOP + BOTTOM) / H_PT])
    return fig, ax


def style(ax, ylabel: str):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=TYPE["tick"], length=2.2, color=GRID, pad=TICK_PAD)
    ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=TYPE["label"], color=INK, labelpad=2.5, linespacing=1.1)
    # five names do not fit on one row of a 190 pt panel at 7 pt: alternate them over two rows,
    # and run the tick of a lower-row name down to it so it reads as belonging to its column
    labels = [l for _, l in BACKBONES]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    drop = 1.25 * TYPE["tick"]
    for i, t in enumerate(ax.xaxis.get_major_ticks()):
        if i % 2:
            t.set_pad(TICK_PAD + drop)
            t.tick1line.set_markersize(2.2 + drop)


def legend(fig, ax, handles, labels, rows=1):
    """legend_above, centred on the panel rather than on its axes.

    The subcaption under the panel is centred on the canvas, and so is the room a legend has:
    centred on the axes, which the axis label pushes right, a 7 pt legend runs off the edge.
    Two rows are filled so they read left to right: matplotlib fills a legend column by column,
    so three entries in two columns are reordered to put (0, 1) on the first row and (2) below.
    """
    if rows == 2:
        order = [0, 2, 1]
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    leg = legend_above(ax, 3 if rows == 1 else 2, handles=handles, labels=labels)
    leg.set_bbox_to_anchor((0.5, 1.02), transform=blended_transform_factory(fig.transFigure,
                                                                             ax.transAxes))
    return leg


def ceiling(root, series, out: Path):
    """What one seed recovers of the closed-set ceiling, on mAP and on Rank-1."""
    maps = [100 * val(root, series, b, "memory", "p2", strategy="random")
            / val(root, series, b, "memory", "p1") for b, _ in BACKBONES]
    r1s = [100 * val(root, series, b, "memory", "p2", "R1", strategy="random")
           / val(root, series, b, "memory", "p1", "R1") for b, _ in BACKBONES]
    x = np.arange(len(BACKBONES)); w = 0.36
    fig, ax = panel()
    h_map = ax.bar(x - w / 2, maps, w, color="#3b6ea5")
    h_r1 = ax.bar(x + w / 2, r1s, w, color="#c4703a")
    h_ceil = ax.axhline(100, color=INK, lw=0.9, ls="--")
    ax.set_ylim(0, 115); ax.set_yticks([0, 50, 100])
    ax.set_xlim(-0.55, len(BACKBONES) - 0.45)
    style(ax, "recovered\n(% of P1)")
    legend(fig, ax, [h_map, h_r1, h_ceil], ["mAP", "Rank-1", "closed-set ceiling"])
    save_fixed(fig, out, pad=None)
    return min(maps), max(maps), min(r1s), max(r1s)


def acceptance(root, series, out: Path):
    """Static gallery, growth by top-1, and growth by true label, at k=1."""
    kw = dict(proto="p4", metric="R1_identity", strategy="random")
    fro = [val(root, series, b, "memory", policy="frozen", **kw) for b, _ in BACKBONES]
    pre = [val(root, series, b, "memory", policy="predicted", **kw) for b, _ in BACKBONES]
    ora = [val(root, series, b, "memory", policy="oracle", **kw) for b, _ in BACKBONES]
    x = np.arange(len(BACKBONES)); w = 0.27
    fig, ax = panel()
    h_fro = ax.bar(x - w, fro, w, color="#8c8c8c")
    h_pre = ax.bar(x, pre, w, color="#c4703a")
    h_ora = ax.bar(x + w, ora, w, color="#3b6ea5")
    for xi, (f, o) in enumerate(zip(fro, ora)):
        ax.annotate("", xy=(xi + w, o), xytext=(xi + w, f),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0,
                                    mutation_scale=5))
        # the gap sits on top of the bar it measures, clear of the neighbouring group
        ax.text(xi + w, o + 2.0, f"+{o - f:.0f}", fontsize=TYPE["annot"], color=INK,
                ha="center", va="bottom")
    ax.set_ylim(0, 75); ax.set_yticks([0, 20, 40, 60])
    ax.set_xlim(-0.55, len(BACKBONES) - 0.45)
    style(ax, "P4 identity\nRank-1")
    legend(fig, ax, [h_fro, h_pre, h_ora], ["static gallery", "grow by top-1", "grow by true label"],
           rows=2)
    save_fixed(fig, out, pad=None)
    return [o - f for f, o in zip(fro, ora)], [p - f for f, p in zip(fro, pre)]


def adaptation(root, series, out: Path):
    """What the released backbone gains in P1 mAP from fine-tuning and from the memory block.

    The memory-block value is the better of `memory` and `memory_lora`. Fine-tuning a BNNeck is
    worth 0.3 to 1.1 points, and the whole point of the panel is that the increments are small.
    """
    pre = [val(root, series, b, "pretrained", "p1") for b, _ in BACKBONES]
    fin = [val(root, series, b, "finetuned", "p1") for b, _ in BACKBONES]
    mem = []
    for b, _ in BACKBONES:
        cand = [val(root, series, b, c, "p1") for c in ("memory", "memory_lora")]
        mem.append(max(v for v in cand if v is not None))
    x = np.arange(len(BACKBONES))
    fig, ax = panel()
    for xi, (a, c) in enumerate(zip(pre, mem)):
        ax.plot([xi, xi], [a, c], color=GRID, lw=1.0, zorder=1)
        # the total sits above the top marker: beside it, it would run into the next column
        ax.annotate(f"+{c - a:.1f}", (xi, c), xytext=(0, 4), textcoords="offset points",
                    fontsize=TYPE["annot"], color=INK, ha="center", va="bottom")
    kw = dict(s=30, zorder=4)
    h_pre = ax.scatter(x, pre, c="#8c8c8c", **kw)
    h_fin = ax.scatter(x, fin, c="#c4703a", **kw)
    h_mem = ax.scatter(x, mem, c="#3b6ea5", **kw)
    lo, hi = min(pre), max(mem)
    ax.set_ylim(lo - 2.0, hi + 6.0)
    ax.set_xlim(-0.55, len(BACKBONES) - 0.45)
    style(ax, "P1 mAP")
    legend(fig, ax, [h_pre, h_fin, h_mem], ["released backbone", "+ BNNeck", "+ memory block"],
           rows=2)
    save_fixed(fig, out, pad=None)
    return [round(c - a, 2) for a, c in zip(pre, mem)], [round(b_ - a, 2) for a, b_ in zip(pre, fin)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "popcharacters")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "generated" / "figures")
    args = ap.parse_args(argv)
    series = load_split()["test"]
    args.out.mkdir(parents=True, exist_ok=True)
    lo_m, hi_m, lo_r, hi_r = ceiling(args.results, series, args.out / "ceiling_recovery.pdf")
    gaps, costs = acceptance(args.results, series, args.out / "p4_closes_gap.pdf")
    tot, ft = adaptation(args.results, series, args.out / "adaptation.pdf")
    print(f"[figures] adaptation.pdf       fine-tuning {min(ft):+.1f} to {max(ft):+.1f}, "
          f"total {min(tot):+.1f} to {max(tot):+.1f}")
    print(f"[figures] ceiling_recovery.pdf  mAP {lo_m:.0f}-{hi_m:.0f} %, Rank-1 {lo_r:.0f}-{hi_r:.0f} %")
    print(f"[figures] p4_closes_gap.pdf     oracle gap +{min(gaps):.1f} to +{max(gaps):.1f}, "
          f"top-1 growth {min(costs):+.1f} to {max(costs):+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
