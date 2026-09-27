"""P3 novelty-threshold sensitivity: every metric the protocol reports, five backbones.

Measured on full per-series streams by `analysis/p3_tau.py` (finetuned backbones, first training
run, no-memory features as P3 specifies, macro over the 8 test series). The figure shows all four
quantities the protocol reports, so the operating point is never read from Purity alone: predicted
clusters per series against the 8.8 true identities, Purity, NMI and ARI. At tau_nov = 0.55 it
reproduces the P3 results of `scripts/evaluate.py`.

    python analysis/p3_tau.py --device cuda \\
        --backbones transreid magiv2 magiv3 instructreid reid5o \\
        --taus 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 \\
        --out results/p3_tau_sweep.json
    python paper/threshold_sensitivity.py --out paper/generated/figures

The four panels print at \\linewidth, so their type is set at the printed sizes of paper_style.TYPE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42    # TrueType: every glyph stays text
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (TYPE, PAGE_W, COLORS, MARKERS, BACKBONES, BACKBONE_LABELS,  # noqa: E402
                         save_fixed)

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "p3_tau_sweep.json"
TRUE_IDS = 8.8           # identities per test series (70 over 8)
REFERENCE = 0.55         # the protocol's fixed rule (src/recognize/protocol_constants.py)
GRID, INK, GREY = "#d9d9d9", "#1b1b1b", "#8c8c8c"
PANELS = [("clusters", "clusters per series", 1.0),
          ("purity", "Purity (%)", 100.0),
          ("nmi", "NMI (%)", 100.0),
          ("ari", r"ARI ($\times$100)", 100.0)]


def sweep(out: Path) -> None:
    data = json.loads(SWEEP.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 3.5), sharex=True)
    for ax, (key, label, scale) in zip(axes.flat, PANELS):
        ax.axvline(REFERENCE, color=GREY, lw=0.7, ls=":", zorder=1)
        for bb in BACKBONES:
            per = data[bb]
            taus = sorted(per, key=float)
            ax.plot([float(t) for t in taus], [scale * per[t][key] for t in taus],
                    color=COLORS[bb], lw=1.0, marker=MARKERS[bb], ms=2.4,
                    label=BACKBONE_LABELS[bb], zorder=3)
        if key == "clusters":
            ax.set_yscale("log")
            ax.axhline(TRUE_IDS, color=GREY, lw=0.8, ls="--", zorder=2)
            ax.text(0.845, TRUE_IDS * 1.12, "true identities", ha="right", va="bottom",
                    fontsize=TYPE["annot"], color=GREY)
            ax.set_yticks([10, 30, 100, 300])
            ax.set_yticklabels(["10", "30", "100", "300"])
            ax.set_ylim(6, 700)
        ax.set_ylabel(label, fontsize=TYPE["label"])
        ax.tick_params(colors=INK, labelsize=TYPE["tick"], length=2.2, color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0.28, 0.87)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    for ax in axes[1]:
        ax.set_xlabel(r"novelty threshold $\tau_{\mathrm{nov}}$", fontsize=TYPE["label"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=TYPE["legend"], handlelength=1.4, handletextpad=0.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    save_fixed(fig, out, pad=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "generated" / "figures")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    sweep(a.out / "threshold_sensitivity.pdf")
    print("[fig] threshold_sensitivity.pdf  tau 0.30-0.85, clusters/Purity/NMI/ARI, 5 backbones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
