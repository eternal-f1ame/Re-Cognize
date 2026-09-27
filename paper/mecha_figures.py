"""The two MeCha appendix diagrams, at the paper's standard full-width shape.

A 3:1 strip suits a pipeline, so both run left to right: the block diagram follows
one crop through to the fused feature, and the inference diagram puts pass 1, pass 2
and the memory write side by side.

Both print at \\linewidth (5.5 in), so a label set at N pt prints at N pt: component
names are at 7.5 pt, notes at 7.2 pt and panel headings at 8 pt, never below
paper_style.FLOOR, and 7.2 pt is the smallest size whose mathtext sub- and
superscripts (0.7 of it) clear paper_style.SCRIPT_FLOOR. Boxes are sized to the text
they hold at those sizes; a note too long for its box is wrapped, never shrunk.

Contents follow the memory block as implemented in src/memory_block/models/: BNNeck
sits before memory, working and episodic memory emit pure deltas, and the only
residual is in the gated fusion.

    python paper/mecha_figures.py --out paper/generated/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42    # TrueType: Greek and accents stay text, not Type 3 XObjects
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import TYPE, fig_size, save_fixed  # noqa: E402

INK, GRID, BLUE, ORANGE, GREY = "#1b1b1b", "#d9d9d9", "#3b6ea5", "#c4703a", "#8c8c8c"
PALE_B, PALE_O, PALE_G = "#dce6f0", "#f2e2d6", "#e2ecdf"
GREEN = "#5b8c62"
UX, UY = 15.0, 5.0                      # one unit is 26.4 pt on both axes
TITLE = TYPE["title"]                   # panel headings
NAME = TYPE["label"]                    # component names
NOTE = 7.2                              # notes; its 0.7 scripts print at 5.0 pt


def box(ax, x, y, w, h, fc="white", ec=GREY, lw=0.9, ls="-", r=0.09, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def label(ax, x, y, t, fs=NOTE, c=INK, ha="center", va="center", weight="normal"):
    ax.text(x, y, t, fontsize=fs, color=c, ha=ha, va=va, weight=weight, zorder=5,
            linespacing=1.15)


def arrow(ax, a, b, color=INK, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=color,
                                 lw=lw, ls=ls, zorder=4, shrinkA=0.5, shrinkB=0.5))


def strip(ax, x, y, w, h, n, colors):
    """A feature vector drawn as n cells."""
    cw = w / n
    for i in range(n):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h, fc=colors[i % len(colors)],
                               ec="white", lw=0.5, zorder=3))


def canvas():
    fig = plt.figure(figsize=fig_size("full"))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, UX); ax.set_ylim(0, UY); ax.axis("off")
    return fig, ax


# ───────────────────────────── the memory block ─────────────────────────────
def architecture(out: Path) -> None:
    fig, ax = canvas()
    row, top, bot = 2.60, 3.28, 1.92     # the pipeline's centre line and box extent

    box(ax, 0.05, bot, 0.90, top - bot, fc=PALE_B, ec=BLUE)
    label(ax, 0.50, 2.86, "crop", NOTE, GREY)
    label(ax, 0.50, 2.36, "$x_t$", NAME)
    arrow(ax, (1.00, row), (1.32, row))

    box(ax, 1.37, bot, 2.70, top - bot, fc=PALE_B, ec=BLUE)
    label(ax, 2.72, 2.95, "frozen backbone", NAME)
    label(ax, 2.72, 2.33, "ViT,\nnever updated", NOTE, GREY)
    arrow(ax, (4.12, row), (4.44, row))

    box(ax, 4.49, bot, 1.40, top - bot, fc=PALE_O, ec=ORANGE)
    label(ax, 5.19, 2.86, "BNNeck", NAME)
    label(ax, 5.19, 2.36, "trained", NOTE, ORANGE)
    strip(ax, 4.48, 1.30, 1.42, 0.32, 6, [BLUE, "#7fa6ce", PALE_B])
    label(ax, 5.19, 0.76, "$\\hat{F}_t$, the\nmemory's input", NOTE, GREY)
    arrow(ax, (5.94, row), (6.40, row))

    # the memory block: two readers, one residual
    fx0, fx1, fy0, fy1 = 6.45, 12.57, 0.20, 4.80
    ax.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, fc="none", ec=GREY, lw=0.9,
                           ls=(0, (3, 2))))
    label(ax, (fx0 + fx1) / 2, 0.56, "memory block", NOTE, GREY)

    wm_x, em_x, bw, by, bh = 6.60, 9.60, 2.82, 2.92, 1.36
    wm_c, em_c = wm_x + bw / 2, em_x + bw / 2
    box(ax, wm_x, by, bw, bh, fc=PALE_G, ec=GREEN)
    label(ax, wm_c, by + bh - 0.30, "working memory", NAME, GREEN)
    label(ax, wm_c, by + 0.42, "$K$ recent\nper identity", NOTE, GREY)
    box(ax, em_x, by, bw, bh, fc=PALE_O, ec=ORANGE)
    label(ax, em_c, by + bh - 0.30, "episodic memory", NAME, ORANGE)
    label(ax, em_c, by + 0.42, "$S$ prototypes\nper identity", NOTE, GREY)

    fu_y, fu_h = 0.94, 1.14
    arrow(ax, (wm_c, by - 0.06), (wm_c, fu_y + fu_h + 0.06), color=GREEN)
    label(ax, wm_c - 0.14, 2.50, r"$\delta_{\mathrm{wm}}$", NAME, GREEN, ha="right")
    arrow(ax, (em_c, by - 0.06), (em_c, fu_y + fu_h + 0.06), color=ORANGE)
    label(ax, em_c + 0.14, 2.50, r"$\delta_{\mathrm{em}}$", NAME, ORANGE, ha="left")

    fu_c = (wm_x + em_x + bw) / 2
    box(ax, wm_x, fu_y, em_x + bw - wm_x, fu_h, fc=PALE_B, ec=BLUE)
    label(ax, fu_c, fu_y + fu_h - 0.30, "gated fusion", NAME, BLUE)
    label(ax, fu_c, fu_y + 0.34,
          r"$\hat{F}_t + \mathrm{proj}(\alpha_{\mathrm{wm}}\delta_{\mathrm{wm}}"
          r" + \alpha_{\mathrm{em}}\delta_{\mathrm{em}})$", NOTE)

    out_c, out_y = 13.72, fu_y + fu_h / 2
    arrow(ax, (fx1 + 0.05, out_y), (13.02, out_y))
    strip(ax, 13.07, out_y - 0.17, 1.30, 0.34, 8, [ORANGE, "#d99a70", PALE_O])
    label(ax, out_c, 2.06, r"$\hat{F}_t^{\,\mathrm{final}}$", NAME)
    label(ax, out_c, 0.98, "L2 normalised", NOTE, GREY)

    # both memories read the same vector the fusion adds back to
    feed_x, feed_y = 6.17, 4.60
    ax.plot([feed_x, feed_x], [row, feed_y], color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.plot([feed_x, em_c], [feed_y, feed_y], color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=2)
    for x in (wm_c, em_c):
        arrow(ax, (x, feed_y), (x, by + bh + 0.04), color=GREY, lw=0.7, ls=(0, (2, 2)))

    save_fixed(fig, out, pad=None)


# ─────────────────────── two-pass open-set inference ───────────────────────
def two_pass(out: Path) -> None:
    fig, ax = canvas()
    y0, y1 = 0.18, 4.82
    panels = [(0.06, 4.10, "pass 1: search all", "no identity is assumed"),
              (4.62, 4.92, "pass 2: routed", "by the identity pass 1 guessed"),
              (10.02, 4.92, "then write", "after the query is answered")]
    for x, w, t, s in panels:
        ax.add_patch(Rectangle((x, y0), w, y1 - y0, fc="none", ec=GRID, lw=0.9))
        label(ax, x + w / 2, 4.46, t, TITLE, INK, weight="bold")
        label(ax, x + w / 2, 4.04, s, NOTE, GREY)

    # pass 1
    c1 = 0.06 + 4.10 / 2
    box(ax, 0.30, 2.82, 3.62, 0.86, fc=PALE_B, ec=BLUE)
    label(ax, c1, 3.25, "episodic memory,\nevery identity", NOTE)
    arrow(ax, (c1, 2.76), (c1, 2.42))
    box(ax, 0.30, 1.52, 3.62, 0.84, fc="white", ec=GREY)
    label(ax, c1, 1.94, r"$\arg\max_c\,\max_s\,\cos(\hat{F}_t, \mu_c^s)$", NOTE)
    arrow(ax, (c1, 1.46), (c1, 1.12))
    box(ax, c1 - 1.02, 0.46, 2.04, 0.60, fc=PALE_O, ec=ORANGE)
    label(ax, c1, 0.76, r"a guess, $\hat{c}_t$", NOTE, ORANGE)
    arrow(ax, (c1 + 1.08, 0.76), (4.86, 0.76), color=ORANGE)

    # pass 2
    c2 = 4.62 + 4.92 / 2
    box(ax, 4.92, 2.94, 1.72, 0.70, fc=PALE_G, ec=GREEN)
    label(ax, 5.78, 3.29, r"WM at $\hat{c}_t$", NOTE, GREEN)
    box(ax, 6.84, 2.94, 2.42, 0.70, fc=PALE_O, ec=ORANGE)
    label(ax, 8.05, 3.29, "EM, all identities", NOTE, ORANGE)
    arrow(ax, (5.78, 2.88), (5.78, 2.44), color=GREEN)
    arrow(ax, (8.05, 2.88), (8.05, 2.44), color=ORANGE)
    box(ax, 4.92, 1.68, 4.34, 0.70, fc=PALE_B, ec=BLUE)
    label(ax, c2, 2.03, "gated fusion", NOTE, BLUE)
    arrow(ax, (c2, 1.62), (c2, 1.12))
    box(ax, c2 - 0.72, 0.44, 1.44, 0.64, fc="white", ec=GREY)
    label(ax, c2, 0.76, r"$\hat{F}_t^{\,\mathrm{final}}$", NOTE)
    arrow(ax, (c2 + 0.78, 0.76), (10.26, 0.76))

    # write
    c3 = 10.02 + 4.92 / 2
    box(ax, 10.28, 2.40, 4.40, 1.28, fc=PALE_G, ec=GREEN)
    label(ax, c3, 3.38, "working memory", NOTE, GREEN)
    label(ax, c3, 2.80, "append to the $\\hat{c}_t$ buffer,\ndrop the oldest", NOTE, GREY)
    box(ax, 10.28, 1.20, 4.40, 0.96, fc=PALE_O, ec=ORANGE)
    label(ax, c3, 1.88, "episodic memory", NOTE, ORANGE)
    label(ax, c3, 1.46, r"replace the nearest slot of $\hat{c}_t$", NOTE, GREY)
    label(ax, c3, 0.76, "neither is a parameter", NOTE, GREY)

    save_fixed(fig, out, pad=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "paper" / "generated" / "figures")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    architecture(a.out / "Architecture-Diagram.pdf")
    two_pass(a.out / "two_pass_open_set_inference_protocol.pdf")
    print("[fig] Architecture-Diagram.pdf, two_pass_open_set_inference_protocol.pdf  3:1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
