"""Shared style for publication-quality figures.

The generators that use it write their figures to paper/generated/figures/.

Usage:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from paper_style import *
    setup_style()
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Embed fonts as TrueType, not matplotlib's default Type 3. Type 3 draws Greek letters and
# accents (tau, delta, hats) as vector paths, so they are not text in the PDF: not searchable,
# and invisible to figure_audit.py, whose size and collision checks then miss them.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ── Layout dimensions (inches) ───────────────────────────────────────
# Two-column dimensions. The paper is single-column (5.5" text width) and its
# figures are sized by the float shapes below, not by these.
COL_WIDTH = 3.4     # two-column layout, one column (≈3.25" after tight bbox)
TEXT_WIDTH = 7.0    # two-column layout, full text width

# ── Standard float shapes ────────────────────────────────────────────
# Every figure float in the paper is one of three shapes: a full-width float is 3:1,
# a full-width float whose content is a single band is 4:1, and a float that takes
# half the text width is 2:1.
#
# The canvas is sized to the width the float actually occupies on the page, so a
# label authored at 7 pt reaches the reader at 7 pt and nothing is scaled. A smaller
# canvas stretched by \includegraphics would scale its type by a different factor in
# every figure.
PAGE_W = 5.5        # \linewidth: 396 pt
HALF_W = 2.64       # a 0.48\linewidth subfigure: 190 pt

# Full-width floats are \includegraphics'd at \linewidth and half-width ones at
# 0.48\linewidth, never scaled further: printing a 5.5" canvas at a smaller width
# shrinks every label below the size it was authored at, so a label set at FLOOR
# prints below it.

# ── Type scale, in printed points ────────────────────────────────────
# Body text is 10 pt and captions 9 pt. Nothing inside a figure prints below FLOOR,
# sub- and superscripts below SCRIPT_FLOOR. paper/figure_audit.py checks every
# figure against both at the width it is printed, and also flags text on text, text
# crossing a box edge and text leaving the canvas.
FLOOR = 7.0
SCRIPT_FLOOR = 5.0
TYPE = {"title": 8.0, "label": 7.5, "body": 7.0, "tick": 7.0, "legend": 7.0, "annot": 7.0}

SHAPES = {
    'full': (PAGE_W, 3.0),
    'band': (PAGE_W, 4.0),
    'half': (HALF_W, 2.0),
}


def fig_size(shape):
    """Canvas size in inches for one of the standard float shapes."""
    w, aspect = SHAPES[shape]
    return (w, w / aspect)


def legend_above(ax, ncol, fontsize=7.0, handles=None, labels=None, gap=0.02):
    """Put the legend on a band above the axes, where the data cannot reach it.

    Inside the plot a legend has to be placed by hand against whatever the data
    happens to do that run, and on a 190 pt panel it loses. Above the axes its
    position is fixed and the axes simply gets shorter; tight_layout reserves the
    band because a legend counts toward the axes' tight bbox.
    """
    kw = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0 + gap), ncol=ncol,
              frameon=False, fontsize=fontsize, handlelength=1.1,
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    if handles is not None:
        return ax.legend(handles, labels, **kw)
    return ax.legend(**kw)


def save_fixed(fig, out, pad=0.3):
    """Save at exactly the canvas size, so the page aspect is the standard one.

    A tight bbox re-crops to the drawn content, which moves the aspect a few
    percent, by a different amount for every figure. Laying out inside the canvas
    keeps the shape exact. Pass pad=None for a figure that already owns its whole
    canvas, such as a schematic drawn on one full-bleed axes.
    """
    if pad is not None:
        fig.tight_layout(pad=pad)
    previous = plt.rcParams['savefig.bbox']
    plt.rcParams['savefig.bbox'] = 'standard'
    try:
        fig.savefig(out)
    finally:
        plt.rcParams['savefig.bbox'] = previous
    plt.close(fig)


# ── Backbone palette: colorblind-safe, print-distinguishable ─────────
COLORS = {
    'transreid':    '#4C72B0',   # steel blue
    'magiv2':       '#DD8452',   # warm orange
    'magiv3':       '#55A868',   # sage green
    'instructreid': '#C44E52',   # muted red
    'reid5o':       '#8172B3',   # muted purple
}

MARKERS = {
    'transreid':    'o',   # circle
    'magiv2':       's',   # square
    'magiv3':       '^',   # triangle up
    'instructreid': 'D',   # diamond
    'reid5o':       'v',   # triangle down
}

BACKBONE_LABELS = {
    'transreid':    'TransReID',
    'magiv2':       'MagiV2',
    'magiv3':       'MagiV3',
    'instructreid': 'InstructReID',
    'reid5o':       'ReID5o',
}

BACKBONES = ['transreid', 'magiv2', 'magiv3', 'instructreid', 'reid5o']

# ── Semantic colors ──────────────────────────────────────────────────
C_BASELINE = '#969696'   # medium gray
C_MECHA    = '#4C72B0'   # steel blue

# Protocol comparison
C_P1 = '#4C72B0'
C_P2 = '#DD8452'
C_P4 = '#55A868'

# Train / val split
C_TRAIN = '#4C72B0'
C_VAL   = '#DD8452'
C_ACCENT = '#55A868'


def setup_style():
    """Apply publication rcParams. Call once per script."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        # ── Sizes tuned for ≈3.25" column width ──
        'font.size': 7,
        'axes.labelsize': 7.5,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        # ── Output quality ──
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        # ── Axes ──
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linewidth': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.5,
        # ── Lines & markers ──
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        # ── Ticks ──
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 2.0,
        'ytick.major.size': 2.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        # ── Legend defaults ──
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.2,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.8,
        'legend.labelspacing': 0.3,
    })


def panel_label(ax, label, x=-0.15, y=1.06):
    """Add bold panel label, e.g. panel_label(ax, 'a')."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left')
