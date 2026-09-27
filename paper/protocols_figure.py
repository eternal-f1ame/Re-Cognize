"""The four RE:COGNIZE protocols, at the paper's standard full-width shape.

P1 and P2 hold the gallery fixed and differ in how much of it exists; P3 starts
empty and admits a crop only past the novelty threshold; P4 starts from seeds and
grows under a bounded buffer.

The gallery entries and the query stream are real POPCharacters crops (Bakuman
chapter 1: Takagi, Mashiro and Azuki, the same characters as the teaser). A frame's
colour is the character's (blue Takagi, orange Mashiro, green Azuki), a dashed
frame marks an entry the protocol appended itself, and the stream is grey because
its labels are what is unknown.

Type follows paper_style.TYPE at the printed size (the canvas is \linewidth, never
scaled): titles 8 pt, every other label 7 pt, and a label carrying a subscript 7.5 pt
so the subscript itself clears SCRIPT_FLOOR. Panel widths follow their content, so
the padding inside each panel is the same even though P2 and P4 hold more.

    python paper/protocols_figure.py --out paper/generated/figures
    python paper/figure_audit.py Protocols.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import TYPE, fig_size, save_fixed  # noqa: E402
import recast_figure_images as images  # noqa: E402

INK, GRID, BLUE, ORANGE, GREY = "#1b1b1b", "#d9d9d9", "#3b6ea5", "#c4703a", "#8c8c8c"
GREEN = "#4f8a5b"
PALE_B, PALE_O = "#dce6f0", "#f2e2d6"

# One full-bleed axes at 15 x 5 data units. The canvas is 3:1 and so is the data
# box, so a unit is 26.4 pt in both directions and nothing is stretched.
UX, UY = 15.0, 5.0
TITLE, BODY = TYPE["title"], TYPE["body"]
MATH = TYPE["label"]           # a label with a subscript: 7.5 pt puts the script at 5.25 pt
CW, CH = 0.54, 0.78            # a gallery crop: 14.3 x 20.6 pt on the page
SW, SH = 0.50, 0.64            # a stream crop
OUT, GAP = 0.12, 0.18          # canvas margin, and the gap between panels

# Vertical bands, bottom to top: the stream, the arrows, the gallery, the title chip.
STREAM = (0.10, 1.36)
GALLERY = (1.98, 4.30)
HEAD = (4.40, 4.88)
LINE1, LINE2 = 4.08, 3.74      # the gallery's one- or two-line description
CROP_Y = 2.38                  # gallery crops sit centred under the description

# Widths of what each panel holds (measured at the sizes above, in data units).
DOTK = 0.50                    # "$\cdots k$" at 7 pt
CHIP_L, CHIP_R, CHIP_GAP = 1.36, 1.37, 0.28   # "assign it", "open a / new one"
PLUS = 0.40
CONTENT = [5 * CW + 4 * 0.10,
           3 * (CW + 0.05 + DOTK) + 2 * 0.14,
           CHIP_L + CHIP_GAP + CHIP_R,
           5 * CW + 3 * 0.06 + PLUS]
PAD = (UX - 2 * OUT - 3 * GAP - sum(CONTENT)) / 8
WIDTHS = [c + 2 * PAD for c in CONTENT]

# Bakuman chapter 1, POPCharacters. Identity A is Takagi (blue), B Mashiro (orange),
# C Azuki (green); a frame is dashed when the protocol appended that entry itself.
S = "Bakuman/Bakuman - c001 (web) - p%03d [Unknown]"
A1, A2, A3 = (S % 17, "Akito Takagi", 0), (S % 25, "Akito Takagi", 0), (S % 18, "Akito Takagi", 1)
B1, B2 = (S % 12, "Moritaka Mashiro", 0), (S % 11, "Moritaka Mashiro", 0)
C1, C2 = (S % 50, "Miho Azuki", 0), (S % 51, "Miho Azuki", 0)
IDCOL = {"Akito Takagi": BLUE, "Moritaka Mashiro": ORANGE, "Miho Azuki": GREEN}
QUERIES = [(S % 4, "Moritaka Mashiro", 0), (S % 7, "Moritaka Mashiro", 0), (S % 8, "Miho Azuki", 0),
           (S % 9, "Moritaka Mashiro", 0), (S % 18, "Moritaka Mashiro", 0), (S % 22, "Akito Takagi", 0),
           (S % 25, "Moritaka Mashiro", 0), (S % 50, "Miho Azuki", 1)]


def chip(ax, x, y, w, h, fc="white", ec=GREY, lw=0.8, ls="-", text=None, fs=BODY, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, fontsize=fs, color=color,
                ha="center", va="center", zorder=4, linespacing=1.25)


def arrow(ax, a, b, color=INK, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6,
                                 color=color, lw=lw, ls=ls, zorder=4, shrinkA=0.5, shrinkB=0.5))


def crop(ax, who, x, y, w=CW, h=CH, ec=None, ls="-", lw=0.9, scale=6):
    """A real crop at its printed size, framed in the colour that carries the semantics.

    Same mechanics as recast_figures.place: the axes is full-bleed, so a data unit is a
    fixed number of points, the crop is rendered at `scale` times that so it survives a
    300 dpi raster, and imshow needs interpolation="none" or matplotlib resamples it to
    the figure's own dpi grid before embedding.
    """
    stem, char, rank = who
    ec = IDCOL[char] if ec is None else ec
    fw, fh = ax.figure.get_size_inches()
    img = images._render(("crop", stem, char, rank),
                         max(8, round(w / UX * fw * 72 * scale)),
                         max(8, round(h / UY * fh * 72 * scale)), scale=1, corpus=images.POP)
    ax.imshow(np.asarray(img), extent=(x, x + w, y, y + h), aspect="auto", zorder=3,
              interpolation="none")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, ls=ls, zorder=4))


def protocols(out: Path) -> None:
    fig = plt.figure(figsize=fig_size("full"))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, UX); ax.set_ylim(0, UY); ax.axis("off")

    # the query stream is the same under all four protocols, so it is drawn once
    s0, s1 = STREAM
    ax.add_patch(Rectangle((OUT, s0), UX - 2 * OUT, s1 - s0, fc="none", ec=GRID, lw=0.8))
    ax.text(0.30, 1.10, "one query stream, in reading order, identical under all four",
            fontsize=BODY, color=GREY, va="center")
    qy = 0.20
    for i in range(9):
        x = 0.30 + i * 1.68
        if i < 8:
            crop(ax, QUERIES[i], x, qy, SW, SH, ec=GREY, lw=0.7)
            ax.text(x + SW + 0.08, qy + SH / 2, r"$q_{%d}$" % (i + 1), fontsize=MATH,
                    color=GREY, va="center")
            arrow(ax, (x + SW + 0.52, qy + SH / 2), (x + 1.58, qy + SH / 2), color=GREY, lw=0.6)
        else:
            ax.text(x + 0.25, qy + SH / 2, r"$\cdots$", fontsize=BODY, color=GREY,
                    va="center", ha="center")

    panels = [
        ("P1", "full gallery",      ("every identity,", "20 % of its crops"), False, "retrieve"),
        ("P2", "$k$ seeded",        ("$k$ crops per identity",),              False, "retrieve"),
        ("P3", "empty",             ("nothing is given",),                    True,  "cluster"),
        ("P4", "$k$ seeded, grows", ("seeds, then appends",
                                     r"drop oldest past $B_{\max}$"),         True,  "retrieve"),
    ]
    g0, g1 = GALLERY
    x0 = OUT
    for i, ((tag, title, sub, grows, verb), W) in enumerate(zip(panels, WIDTHS)):
        accent = ORANGE if grows else BLUE
        pale = PALE_O if grows else PALE_B
        left = x0 + PAD                                  # where the panel's content starts

        chip(ax, x0, HEAD[0], W, HEAD[1] - HEAD[0], fc=pale, ec=accent, lw=0.9)
        hy = (HEAD[0] + HEAD[1]) / 2
        ax.text(x0 + 0.14, hy, tag, fontsize=TITLE, color=accent, weight="bold",
                va="center", zorder=4)
        ax.text(x0 + 0.74, hy, title, fontsize=TITLE, color=INK, va="center", zorder=4)

        ax.add_patch(Rectangle((x0, g0), W, g1 - g0, fc="none", ec=accent, lw=0.9,
                               ls=(0, (2.5, 1.8)) if grows else "-"))
        for line, (y, text) in enumerate(zip((LINE1, LINE2), sub)):
            # P4's second line is the buffer rule, in the colour of growth
            ax.text(x0 + W / 2, y, text, ha="center", va="center",
                    fontsize=MATH if "$B" in text else BODY,
                    color=ORANGE if line == 1 and grows else GREY)

        if i == 0:
            for j, who in enumerate((A1, A2, B1, B2, C1)):
                crop(ax, who, left + j * (CW + 0.10), CROP_Y)
        elif i == 1:
            for j, who in enumerate((A1, B1, C1)):
                x = left + j * (CW + 0.05 + DOTK + 0.14)
                crop(ax, who, x, CROP_Y)
                ax.text(x + CW + 0.05, CROP_Y + CH / 2, r"$\cdots k$", fontsize=BODY,
                        color=GREY, va="center")
        elif i == 2:
            cx = x0 + W / 2
            ax.text(cx, 3.64, r"$\max\,\mathrm{sim} \geq \tau_{\mathrm{nov}}$?",
                    fontsize=MATH, color=INK, va="center", ha="center")
            cy, ch = 2.20, 0.80
            lx, rx = left, left + CHIP_L + CHIP_GAP
            arrow(ax, (cx - 0.12, 3.40), (lx + CHIP_L / 2 + 0.10, cy + ch + 0.05), color=BLUE, lw=0.7)
            arrow(ax, (cx + 0.12, 3.40), (rx + CHIP_R / 2 - 0.10, cy + ch + 0.05), color=ORANGE, lw=0.7)
            chip(ax, lx, cy, CHIP_L, ch, fc=PALE_B, ec=BLUE, text="assign it")
            chip(ax, rx, cy, CHIP_R, ch, fc=PALE_O, ec=ORANGE, text="open a\nnew one")
        else:
            for j, who in enumerate((A1, B1, C1)):
                crop(ax, who, left + j * (CW + 0.06), CROP_Y)
            px = left + 3 * CW + 2 * 0.06
            ax.text(px + PLUS / 2, CROP_Y + CH / 2, "+", fontsize=BODY + 1, color=ORANGE,
                    ha="center", va="center")
            for j, who in enumerate((A3, C2)):
                crop(ax, who, px + PLUS + j * (CW + 0.06), CROP_Y, ls=(0, (2, 1.6)))

        # gallery to stream, and for the two protocols that grow, stream to gallery
        ay0, ay1 = s1 + 0.05, g0 - 0.05
        arrow(ax, (x0 + 0.42, ay1), (x0 + 0.42, ay0), color=INK)
        ax.text(x0 + 0.56, (ay0 + ay1) / 2, verb, fontsize=BODY, color=INK, va="center")
        if grows:
            arrow(ax, (x0 + W - 0.42, ay0), (x0 + W - 0.42, ay1), color=ORANGE)
            ax.text(x0 + W - 0.56, (ay0 + ay1) / 2, "append", fontsize=BODY, color=ORANGE,
                    va="center", ha="right")
        x0 += W + GAP

    save_fixed(fig, out, pad=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "paper" / "generated" / "figures")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    protocols(a.out / "Protocols.pdf")
    print("[fig] Protocols.pdf  four panels, 3:1, real crops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
