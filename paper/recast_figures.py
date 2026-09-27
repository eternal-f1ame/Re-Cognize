"""Figures for the Re:Cast sections of the paper, in the style of its other main-text plots.

Type follows paper_style.TYPE at the printed size: the two schematics are drawn on
canvases of \\linewidth and the two plots on 0.48\\linewidth, and the paper includes
them at exactly that width, so a label authored at 7 pt prints at 7 pt.
expansion_floor.pdf is not in the paper and is not held to those sizes.

    python paper/recast_figures.py --out paper/generated/figures
    python paper/recast_figures.py --only recast_schematic_a recast_schematic_b \\
        one_breakeven supervision_geometry
    python paper/figure_audit.py recast_schematic_a.pdf recast_schematic_b.pdf \\
        one_breakeven.pdf supervision_geometry.pdf
"""
from __future__ import annotations

import argparse, glob, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from paper_style import TYPE, fig_size, save_fixed, legend_above
import recast_figure_images as images

INK, GRID, BLUE, ORANGE, GREY = "#1b1b1b", "#d9d9d9", "#3b6ea5", "#c4703a", "#8c8c8c"
PALE_B, PALE_O = "#dce6f0", "#f2e2d6"


def style(ax, tick=5.5):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=tick, length=2.2, color=GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)


def crop(ax, x, y, w=0.34, h=0.44, fc="white", ec=GREY, lw=0.9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))


def arrow(ax, a, b, color=INK, lw=0.9, style_="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style_, mutation_scale=8,
                                 color=color, lw=lw, ls=ls, zorder=4,
                                 shrinkA=1, shrinkB=1))


def legend_rows(fig, rows, gap_pt=1.5):
    """Legend rows stacked on a band above the axes, centred on the canvas.

    legend_above centres on the axes, which on a 190 pt panel sits right of the canvas
    centre by half the y label, so a legend as wide as the panel runs into the right
    edge. These are figure legends with the same 7 pt styling, one per row (so a row can
    hold a label too long to share one), and the return value is the top of the space
    left for the axes, to pass to tight_layout as its rect.
    """
    H = fig.get_size_inches()[1] * 72
    y = 1 - gap_pt / H
    for handles, labels in rows:
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                         ncol=len(handles), frameon=False, fontsize=TYPE["legend"],
                         handlelength=1.1, handletextpad=0.4, columnspacing=1.2,
                         borderaxespad=0.0, borderpad=0.2)
        fig.canvas.draw()
        y = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    return y - gap_pt / H


def place(ax, spec, x, y, w, h, z=3, scale=6):
    """Drop a Re:Verse page or crop into a data rectangle, at its printed size.

    The axes is full-bleed, so its data box maps to the whole canvas and a unit is a
    fixed number of points. `scale` oversamples against that so the page survives a
    300 dpi raster, and imshow needs interpolation="none" or matplotlib resamples the
    array down to the figure's own dpi grid before embedding it.
    """
    fw, fh = ax.figure.get_size_inches()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    img = images._render(spec, max(8, round(w / (x1 - x0) * fw * 72 * scale)),
                         max(8, round(h / (y1 - y0) * fh * 72 * scale)),
                         scale=1, corpus=images.RV)
    ax.imshow(np.asarray(img), extent=(x, x + w, y, y + h), aspect="auto", zorder=z,
              interpolation="none")


# ────────────────────────────── A. schematic, three moves ──────────────────────────────
def schematic_three(out: Path):
    """The three moves Re:Cast makes, on real pages.

    Crops and pages are Re:Verse's Re:Zero. Identity A is Rom throughout, so the
    same character carries the cast sheet, the commitment and the seed expansion.

    3:1, not the 4:1 band: at 7 pt the three columns need the full 396 pt of width
    between them (b's rule alone is 134 pt), and c's chain from the seed's page to the
    mean it yields is five rows deep, which a 99 pt band cannot hold without crowding.
    One data unit is 26.4 pt in both directions.
    """
    fig = plt.figure(figsize=fig_size("full"))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 15); ax.set_ylim(0, 5)
    B, T = TYPE["body"], TYPE["title"]

    def head(x, t, sub):
        ax.text(x, 4.72, t, fontsize=T, color=INK, weight="bold", va="center")
        ax.text(x, 4.34, sub, fontsize=B, color=GREY, va="center")

    def dot(x, y, c):
        ax.scatter([x], [y], s=46, c=c, edgecolors="none", zorder=4)

    # (a) a bag of exemplars per identity collapses to one vector each
    head(0.15, "a. cast sheet", "one vector per identity")
    bag = [(BLUE, [("004_0025", "Rom", r) for r in (0, 2, 3)]),
           (ORANGE, [("008_0013", "Subaru", 0), ("001_0011", "Subaru", 1),
                     ("001_0009", "Subaru", 0)]),
           (GREY, [("008_0013", "Felt", r) for r in (0, 1, 2)])]
    rows = (2.86, 1.90, 0.94)
    cw, ch = 0.54, 0.76
    for row, (c, specs) in enumerate(bag):
        for col, (stem, who, rank) in enumerate(specs):
            x = 0.18 + col * 0.59
            place(ax, ("crop", stem, who, rank), x, rows[row], cw, ch)
            ax.add_patch(Rectangle((x, rows[row]), cw, ch, fc="none", ec=c, lw=0.9, zorder=5))
    mid = rows[1] + ch / 2
    arrow(ax, (2.06, mid), (2.86, mid))
    ax.text(2.46, mid + 0.22, "mean", fontsize=B, color=INK, ha="center", va="center")
    for i, (y, c) in enumerate(zip(rows, (BLUE, ORANGE, GREY))):
        dot(3.08, y + ch / 2, c)
        ax.text(3.26, y + ch / 2, f"identity {chr(65 + i)}", fontsize=B, color=INK, va="center")

    # (b) a crop joins only where its own page already names the identity
    bx = 4.92
    head(bx, "b. commitment", "only where the page already names it")
    px, py, pw, ph = bx, 1.06, 1.66, 2.40
    place(ax, ("page", "004_0029", {"Rom": BLUE}), px, py, pw, ph)
    ax.add_patch(Rectangle((px, py), pw, ph, fc="none", ec=GRID, lw=0.9, zorder=5))
    ax.text(px + pw / 2, 3.72, "one page", fontsize=B, color=GREY, ha="center", va="center")
    cx, cw2, ch2 = px + pw + 0.52, 0.62, 0.76
    top, bot = 2.70, 1.72                       # the named crop, and its sibling below it
    for y, rank, ls in ((top, 0, "-"), (bot, 1, (0, (2, 1.6)))):
        place(ax, ("crop", "004_0029", "Rom", rank), cx, y, cw2, ch2)
        ax.add_patch(Rectangle((cx, y), cw2, ch2, fc="none", ec=BLUE, lw=1.2, ls=ls, zorder=5))
    ax.text(cx + cw2 / 2, 3.72, "named", fontsize=B, color=BLUE, ha="center", va="center")
    ax.text(cx + cw2 / 2, bot - 0.27, "its sibling", fontsize=B, color=GREY, ha="center",
            va="center")
    ay = (top + bot + ch2) / 2
    a0, a1 = cx + cw2 + 0.16, cx + cw2 + 1.16
    arrow(ax, (a0, ay), (a1, ay), color=BLUE)
    ax.text((a0 + a1) / 2 + 0.04, ay + 0.24, "commit", fontsize=B, color=BLUE, ha="center",
            va="center")
    dot(a1 + 0.24, ay, BLUE)
    ax.text(a1 + 0.44, ay, "identity A\nupdated", fontsize=B, color=INK, va="center",
            linespacing=1.25)
    ax.text(bx, 0.56, "no named crop on the page, no append", fontsize=B, color=GREY,
            va="center")

    # (c) before the stream, a seed's page group joins under its label. The two
    # dashed crops are the expansion, so its yield is printed under them, and the
    # chain to the mean hangs off the seed, which is what now carries it.
    cx0 = 10.86
    head(cx0, "c. seed expansion", "before the stream starts")
    bx0, by0, bw, bh = cx0, 2.66, 3.30, 1.02
    ax.add_patch(Rectangle((bx0, by0), bw, bh, fc="none", ec=GRID, lw=0.9))
    ax.text(bx0 + bw / 2, 3.90, "the seed's own page", fontsize=B, color=GREY, ha="center",
            va="center")
    xs = [bx0 + 0.18 + i * 1.08 for i in range(3)]
    cy = by0 + (bh - ch2) / 2
    for i, x in enumerate(xs):
        place(ax, ("crop", "004_0020", "Rom", i), x, cy, cw2, ch2)
        ax.add_patch(Rectangle((x, cy), cw2, ch2, fc="none", ec=BLUE, zorder=5,
                               lw=1.4 if i == 0 else 0.9,
                               ls="-" if i == 0 else (0, (2, 1.6))))
        if i:
            ax.plot([xs[i - 1] + cw2, x], [cy + ch2 / 2] * 2, color=BLUE, lw=1.0, zorder=4)
    sx = xs[0] + cw2 / 2
    ax.text(sx, by0 - 0.26, "seed", fontsize=B, color=BLUE, ha="center", va="center")
    ax.text(xs[1] - 0.26, by0 - 0.40, "+1.43 crops per seed,\n90.6 % correct", fontsize=B,
            color=INK, va="center", linespacing=1.25)
    arrow(ax, (sx, by0 - 0.50), (sx, 1.40), color=BLUE)
    dot(sx, 1.20, BLUE)
    ax.text(sx + 0.22, 1.20, "a mean of 2.4, not of 1", fontsize=B, color=INK, va="center")

    save_fixed(fig, out, pad=None)


# ────────────────────────── A2. schematic, gallery over time ──────────────────────────
def schematic_timeline(out: Path):
    """Re:Cast over six real pages: it writes only where the page names an identity.

    Pages are Re:Verse's Re:Zero, 005_0005 to 005_0010 in reading order, and whether
    each one counts as an anchor is read from its annotations rather than asserted,
    so the figure cannot disagree with the data it is drawn from. The seed and the
    cast sheet are not repeated here; the row above this one is where they are shown.

    A 4:1 band laid out in printed points (the data box is 396 x 99), top to bottom:
    page labels, pages, what each page does, the timeline, and the sentence.
    """
    ANCHOR = "Rom"
    RUN = [f"005_{n:04d}" for n in range(5, 11)]
    fig = plt.figure(figsize=fig_size("band"))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    X, Y = fig_size("band")[0] * 72, fig_size("band")[1] * 72
    ax.set_xlim(0, X); ax.set_ylim(0, Y)
    B = TYPE["body"]

    # Row centres, in points. A 7 pt line is 7.2 pt tall, a dot 5.8 pt across, and
    # every gap between rows below is at least 3 pt.
    LABEL_Y, TEXT_Y, LINE_Y, R = Y - 6.2, 6.0, 16.6, 2.9
    TOP = LABEL_Y - 6.6                                # page top, under its label
    H = 45.0; W = round(H * 0.695, 1)                  # the pages' own aspect
    BOT = TOP - H
    X0, X1 = 8.0, X - 8.0
    pitch = (X1 - 14.0 - X0 - W) / 5                  # leave the arrowhead its own room

    ax.annotate("", xy=(X1, LINE_Y), xytext=(X0 - 2, LINE_Y),
                arrowprops=dict(arrowstyle="-|>", color=GRID, lw=1.2, mutation_scale=9))
    ax.text(X1, TEXT_Y, "reading order", fontsize=B, color=GREY, ha="right", va="center")

    for n, stem in enumerate(RUN):
        x = X0 + n * pitch
        cx = x + W / 2
        on_page = {nm for nm, *_ in images._boxes(images._stem(stem))}
        hit = ANCHOR in on_page
        place(ax, ("page", stem, {ANCHOR: BLUE} if hit else {}), x, BOT, W, H)
        ax.add_patch(Rectangle((x, BOT), W, H, fc="none", ec=GRID, lw=0.9, zorder=5))
        ax.text(cx, LABEL_Y, f"page {n + 1}", fontsize=B, color=GREY, ha="center", va="center")
        zone = (BOT + LINE_Y + R) / 2                  # between the page and its dot
        if hit:
            arrow(ax, (cx, BOT - 1.5), (cx, LINE_Y + R + 1.0), color=BLUE)
            ax.text(cx + 3.5, zone, "commit", fontsize=B, color=BLUE, va="center")
        else:
            ax.text(cx, zone, "no anchor,\nno append", fontsize=B, color=GREY,
                    ha="center", va="center", linespacing=1.2)
        ax.scatter([cx], [LINE_Y], s=(2 * R) ** 2, c=BLUE if hit else "white",
                   edgecolors=BLUE if hit else GRID, lw=0.9, zorder=6)

    ax.text(X0 + 2.5 * pitch + W / 2, TEXT_Y,
            "the cast sheet is updated only where the page names it",
            fontsize=B, color=INK, ha="center", va="center")
    save_fixed(fig, out, pad=None)


# ─────────────────────── B. supervision geometry, Seq-R against Seq-T ───────────────────────
def supervision(out: Path):
    from recognize.data import SeriesStream, load_split
    from recognize.protocols import split_seeds
    W = 20
    res = {}
    for strat in ("random", "temporal"):
        Q = [[], [], [], []]
        for name in load_split()["test"]:
            s = SeriesStream(ROOT / "Datasets/popcharacters" / name)
            labels, order = np.asarray(s.labels), s.reading_order
            sm, queries, go = split_seeds(labels, order, 5, strat, 0)
            gal = {i for v in sm.values() for i in v} | {i for v in go.values() for i in v}
            qs = set(queries); n = len(order)
            for i, c in enumerate(order):
                if c not in qs: continue
                win = order[max(0, i - W):i]
                Q[min(3, 4 * i // n)].append(sum(1 for x in win if x in gal) / max(1, len(win)))
        res[strat] = [100 * st.mean(q) for q in Q]
    x = np.arange(4); w = 0.36
    fig, ax = plt.subplots(figsize=fig_size("half"))
    ax.bar(x - w / 2, res["random"], w, color=BLUE, label="Seq-R (random seeds)")
    ax.bar(x + w / 2, res["temporal"], w, color=ORANGE, label="Seq-T (chronological)")
    for xi, v in enumerate(res["temporal"]):
        ax.text(xi + w / 2, v + 0.6, f"{v:.1f}", fontsize=TYPE["annot"], ha="center",
                va="bottom", color=ORANGE)
    # headroom over the tallest bar, so its 7 pt value label sits inside the axes
    ax.set_ylim(0, 1.36 * max(res["temporal"] + res["random"]))
    ax.set_yticks([0, 10, 20])
    ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("quarter of the stream", fontsize=TYPE["label"], labelpad=2)
    # two lines: on one, the label is longer than the axes and reaches the legend
    ax.set_ylabel("labelled\nshare (%)", fontsize=TYPE["label"], labelpad=2, linespacing=1.15)
    style(ax, tick=TYPE["tick"])
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(pad=0.45, rect=(0, 0, 1, legend_rows(fig, [(h, l)])))
    save_fixed(fig, out, pad=None)
    return res


# ───────────────── C. one break-even, three mechanisms ─────────────────
#
# Why there is no break-even tick on the x axis. The identity of the paper's Section 5 is
# Delta = c (p_eff - a+), so an append pays exactly while p_eff exceeds a+, the static
# gallery's accuracy on the queries that append captures. The x axis here is a, the
# same gallery's accuracy over the whole query set, which is what an operator can read
# before deciding. Over the fourteen append cells a and a+ differ by up to 3.52 points
# and the margin they give disagrees in sign on MagiV2 / POPCharacters, the cell
# Section 5 uses as its worked exception: p_eff - a = +4.46 there against
# p_eff - a+ = +0.94, and the measured change is -0.04. A p_eff tick on this axis
# would put that cell on the wrong side of it.
#
# p_eff is also not one number per operation. Across the seven must-link cells it runs
# 30.9 to 54.4 capture-weighted (results/commit_*.json, the values
# paper/generated/tables/commit_condition.tex prints) and across the seven top-1 cells 20.8 to 64.5.
# And the identity is exact per series and per gallery seed, not on a cell's plain
# means: mean(c (p_eff - a+)) reproduces the measured change to floating point in all
# seven cells, while mean(c) (mean(p_eff) - mean(a+)) with plain means gives about +0.3
# for MagiV2 / POPCharacters against a measured -0.04, which is why the table prints
# p_eff and a+ capture-weighted (sum c_i x_i / sum c_i), under which the row multiplies out.
#
# The third operation adds nothing to the gallery and obeys a different identity,
# Delta = l (r+ - a+) - (1 - l) a-, whose terms on disk (results/live_pop_*.json and
# live_m109_*.json: ell, gain_on_live, lost_off_live) carry no p_eff at all. So the zero line is the only
# break-even this figure can draw from what is on disk. It is drawn and labelled.
def breakeven(out: Path):
    pts = {"append under the page constraint": ([], [], BLUE, "o", True),
           "append by top-1": ([], [], ORANGE, "s", True),
           "restrict the candidates": ([], [], GREY, "^", False)}
    for f in sorted(glob.glob(str(ROOT / "results/commit_*.json"))):
        d = json.load(open(f)); bb = [k for k in d][0]
        for pol, key in (("mustlink", "append under the page constraint"),
                         ("predicted", "append by top-1")):
            rs = [r for v in d[bb]["by_series"].values() for r in v[pol]]
            pts[key][0].append(100 * st.mean(r["a"] for r in rs))
            pts[key][1].append(100 * st.mean(r["delta"] for r in rs))
    for f in sorted(glob.glob(str(ROOT / "results/live_pop_*.json")) + glob.glob(str(ROOT / "results/live_m109_*.json"))):
        d = json.load(open(f)); bb = [k for k in d][0]
        s = sorted(d[bb])
        g = lambda a: st.mean(st.mean(v["R1_identity"] for v in d[bb][x][a].values()) for x in s)
        pts["restrict the candidates"][0].append(100 * g("static/exemplar"))
        pts["restrict the candidates"][1].append(
            100 * (g("live20/predicted/exemplar") - g("static/exemplar")))
    fig, ax = plt.subplots(figsize=fig_size("half"))
    ax.set_xlim(13, 69); ax.set_ylim(-9.0, 9.8)
    # the losing half, so the sign of a point is readable without tracing the axis
    ax.axhspan(-9.0, 0, color=GREY, alpha=0.07, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=1.0, ls=(0, (3.2, 2.0)), zorder=2)
    ax.text(68.4, 0.7, "break-even", fontsize=TYPE["annot"], color=INK, ha="right",
            va="bottom", zorder=6)
    handles = []
    for lab, (xs, ys, c, m, filled) in pts.items():
        handles.append(ax.scatter(xs, ys, s=25, marker=m, lw=1.0, zorder=5, label=lab,
                                  facecolors=c if filled else "none", edgecolors=c,
                                  alpha=0.85 if filled else 1.0))
    ax.set_yticks([-5, 0, 5])
    ax.set_xlabel("accuracy of the gallery it starts from", fontsize=TYPE["label"], labelpad=2)
    ax.set_ylabel("change in\nRank-1", fontsize=TYPE["label"], labelpad=2, linespacing=1.15)
    style(ax, tick=TYPE["tick"])
    # At 7 pt the first label is 121 pt and the panel 190 pt, so no two-column legend
    # fits. It gets a row of its own, and the other two share the row beneath it.
    labels = list(pts)
    top = legend_rows(fig, [(handles[:1], labels[:1]), (handles[1:], labels[1:])])
    fig.tight_layout(pad=0.45, rect=(0, 0, 1, top))
    save_fixed(fig, out, pad=None)
    return {k: len(v[0]) for k, v in pts.items()}


FIGURES = ("recast_schematic_a", "recast_schematic_b", "supervision_geometry",
           "one_breakeven", "expansion_floor")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "generated" / "figures")
    ap.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES),
                    help="write only these figures (default: all five)")
    a = ap.parse_args(argv); a.out.mkdir(parents=True, exist_ok=True)
    make = {"recast_schematic_a": schematic_three, "recast_schematic_b": schematic_timeline,
            "supervision_geometry": supervision, "one_breakeven": breakeven,
            "expansion_floor": expansion}
    for name in FIGURES:
        if name in a.only:
            r = make[name](a.out / f"{name}.pdf")
            print(f"[fig] {name}.pdf", "" if r is None else r)
    return 0




# ─────────────── D. the k=1 floor, and what expansion does to it ───────────────
def expansion(out: Path):
    BB = [("transreid", "TransReID"), ("magiv2", "MagiV2"), ("magiv3", "MagiV3"),
          ("instructreid", "InstructReID"), ("reid5o", "ReID5o")]
    d = json.load(open(ROOT / "results/expand_k1.json"))
    plain, expd = [], []
    for b, _ in BB:
        s = sorted(d[b])
        g = lambda a: st.mean(st.mean(v["R1_identity"] for v in d[b][x][a].values()) for x in s)
        base = g("exemplar/static")
        plain.append(100 * (g("recast") - base)); expd.append(100 * (g("recast+expand") - base))
    x = np.arange(len(BB))
    fig, ax = plt.subplots(figsize=fig_size("half"))
    ax.axhline(0, color=INK, lw=0.9, ls="--")
    for xi, (lo, hi) in enumerate(zip(plain, expd)):
        ax.annotate("", xy=(xi, hi), xytext=(xi, lo),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3))
        ax.text(xi + 0.13, hi, f"{hi:+.1f}", fontsize=5, color=BLUE, va="center")
    ax.scatter(x, plain, s=34, c=GREY, zorder=5, label="Re:Cast alone, exactly $+0.00$")
    ax.scatter(x, expd, s=34, c=BLUE, zorder=5, label="with seed expansion")
    ax.set_xlim(-0.5, len(BB) - 0.25); ax.set_ylim(-2.1, 8.2)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in BB], rotation=30, ha="right")
    ax.set_ylabel("gain at $k{=}1$", fontsize=6)
    legend_above(ax, 2)
    style(ax); save_fixed(fig, out)
    return list(zip([l for _, l in BB], [round(v, 2) for v in plain], [round(v, 2) for v in expd]))


if __name__ == "__main__":
    sys.exit(main())
