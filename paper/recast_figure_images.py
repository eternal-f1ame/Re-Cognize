"""Fill the image frames of the paper's layout figures with real pages and crops.

The binding and commit figures (Re:Verse) and the teaser (POPCharacters) start from an
exported layout PDF with placeholder frames: it carries the layout, arrows and labels,
and leaves grey frames where imagery belongs. FIGURES below is the single source of
truth for those rectangles, as (left, top, width, height) in layout px: --emit-canva
prints the insert_shape operations that draw them in the layout, and --composite drops
the real pages and crops into the exported PDF, converting px to pt with PX_TO_PT.
protocols_figure.py and recast_figures.py reuse `_render` to place crops in their own
figures.

    python paper/recast_figure_images.py --emit-canva PAGE_ID [--figure commit]
    python paper/recast_figure_images.py --composite raw.pdf out.pdf [--figure commit]
"""
from __future__ import annotations
import argparse, io, json, pathlib, sys
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
RV = ROOT / "Datasets/Re-Verse/Re-Zero"
POP = ROOT / "Datasets/popcharacters"

# Each figure names the corpus its frames are drawn from. Re:Verse keeps one flat
# series, so a stem is "008_0013"; POPCharacters is one directory per series, so a
# stem is "Bakuman/Bakuman - c001 (web) - p031 [Unknown]".
CORPUS = {"binding": RV, "commit": RV, "teaser": POP}
PX_TO_PT = 0.75                      # the layout PDF exports 1600 layout px as 1200 pt
INK, GRID, BLUE, ORANGE, GREY = "#1b1b1b", "#d9d9d9", "#3b6ea5", "#c4703a", "#8c8c8c"
RGB = {BLUE: (59, 110, 165), ORANGE: (196, 112, 58), GREY: (150, 150, 150)}

# figure -> {name: (left, top, width, height, source, stroke)}
# name: (left, top, width, height, source, stroke)
#   source ("page", stem, {char: colour})  -> full page with boxes drawn on it
#   source ("page", stem, None)            -> full page, clean
#   source ("crop", stem, char, rank)      -> rank-th largest box of char on that page
BINDING = {
    # (a) one page with its own boxes drawn on it, and the two clusters those crops form
    "A_PAGE": (40, 96, 252, 364, ("page", "008_0013", {"Subaru": BLUE, "Felt": ORANGE}), GREY),
    "A_S1":   (310, 105, 96, 182, ("crop", "008_0013", "Subaru", 0), BLUE),
    "A_S2":   (406, 105, 96, 182, ("crop", "008_0013", "Subaru", 1), BLUE),
    "A_F1":   (310, 318, 96, 182, ("crop", "008_0013", "Felt", 0), ORANGE),
    "A_F2":   (406, 318, 96, 182, ("crop", "008_0013", "Felt", 1), ORANGE),

    # (b) three pages in true reading order; Subaru recurs and merges, Felt opens
    "B_P1":   (585, 118, 104, 150, ("page", "001_0013", None), GREY),
    "B_P2":   (758, 118, 104, 150, ("page", "001_0014", None), GREY),
    "B_P3":   (931, 118, 104, 150, ("page", "001_0015", None), GREY),
    "B_C1":   (593, 280, 88, 106, ("crop", "001_0013", "Subaru", 0), BLUE),
    "B_C2":   (766, 280, 88, 106, ("crop", "001_0014", "Subaru", 0), BLUE),
    "B_C3":   (939, 280, 88, 106, ("crop", "001_0015", "Felt", 0), ORANGE),
    "B_G1":   (603, 419, 78, 83, ("crop", "001_0013", "Subaru", 0), BLUE),
    "B_G2":   (828, 419, 78, 83, ("crop", "001_0015", "Felt", 0), ORANGE),

    # (c) a seed names a cluster, and all of it is filed under that identity
    "C_SEED": (1105, 96, 150, 180, ("crop", "008_0013", "Subaru", 2), BLUE),
    "C_1":    (1313, 106, 80, 160, ("crop", "011_0018", "Subaru", 0), GREY),
    "C_2":    (1401, 106, 80, 160, ("crop", "004_0024", "Subaru", 0), GREY),
    "C_3":    (1489, 106, 80, 160, ("crop", "003_0019", "Subaru", 0), GREY),
    "C_G1":   (1114, 355, 104, 139, ("crop", "008_0013", "Subaru", 2), BLUE),
    "C_G2":   (1230, 355, 104, 139, ("crop", "011_0018", "Subaru", 0), BLUE),
    "C_G3":   (1346, 355, 104, 139, ("crop", "004_0024", "Subaru", 0), BLUE),
    "C_G4":   (1462, 355, 104, 139, ("crop", "003_0019", "Subaru", 0), BLUE),
}

# the three crops the binding carries into the gallery, inside the "append" stage
COMMIT = {
    "B_SUB":  (62, 82, 60, 72, ("crop", "008_0013", "Subaru", 1), BLUE),
    "B_FELT": (62, 170, 60, 72, ("crop", "008_0013", "Felt", 1), ORANGE),
    "T_1": (648, 109, 62, 104, ("crop", "001_0013", "Subaru", 0), BLUE),
    "T_2": (718, 109, 62, 104, ("crop", "001_0014", "Subaru", 0), BLUE),
    "T_3": (788, 109, 62, 104, ("crop", "001_0015", "Felt", 0), ORANGE),
}


# The page-one teaser, on POPCharacters, the corpus the paper reports on. Three
# beats, matching its caption: a page someone asks a question about, the fixed
# gallery closed-set Re-ID assumes, and the stream that has to build one.
TEASER = {
    # Every panel sits on one grid: heading at 12, sub-heading at 50, visuals from
    # 96 to 452, caption at 468. The rectangles below are the visual band's share.
    "A_PAGE": (34, 166, 198, 286, ("page", "Bakuman/Bakuman - c001 (web) - p018 [Unknown]", {"Akito Takagi": BLUE, "Moritaka Mashiro": ORANGE}), GREY),
    "A_C1":   (250, 166, 132, 136, ("crop", "Bakuman/Bakuman - c001 (web) - p018 [Unknown]", "Akito Takagi", 1), BLUE),
    "A_C2":   (250, 316, 132, 136, ("crop", "Bakuman/Bakuman - c001 (web) - p018 [Unknown]", "Moritaka Mashiro", 0), ORANGE),

    "B_G1":   (498, 142, 96, 116, ("crop", "Bakuman/Bakuman - c001 (web) - p011 [Unknown]", "Moritaka Mashiro", 0), GREY),
    "B_G2":   (610, 142, 96, 116, ("crop", "Bakuman/Bakuman - c001 (web) - p025 [Unknown]", "Akito Takagi", 0), GREY),
    "B_G3":   (722, 142, 96, 116, ("crop", "Bakuman/Bakuman - c001 (web) - p051 [Unknown]", "Miho Azuki", 0), GREY),
    "B_Q1":   (540, 316, 104, 136, ("crop", "Bakuman/Bakuman - c001 (web) - p004 [Unknown]", "Moritaka Mashiro", 0), BLUE),
    "B_Q2":   (680, 316, 104, 136, ("crop", "Bakuman/Bakuman - c001 (web) - p050 [Unknown]", "Miho Azuki", 0), ORANGE),
}

FIGURES = {"binding": BINDING, "commit": COMMIT, "teaser": TEASER}


def _series(short: str, corpus: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Split a stem into the directory holding it and the file stem itself."""
    if "/" in short:
        series, stem = short.rsplit("/", 1)
        return corpus / series, stem
    return corpus, short


def _names(base: pathlib.Path) -> dict[str, str]:
    return json.loads((base / "category_mapping.json").read_text())


def _stem(short: str, corpus: pathlib.Path = RV) -> str:
    base, stem = _series(short, corpus)
    if (base / "annotations" / f"{stem}.txt").exists():
        return stem
    hits = sorted((base / "annotations").glob(stem + "_png*.txt"))
    if not hits:
        sys.exit(f"no annotation for {short}")
    return hits[0].stem


def _boxes(stem: str, corpus: pathlib.Path = RV, series: str = ""):
    base = corpus / series if series else corpus
    names = _names(base)
    out = []
    for line in (base / "annotations" / f"{stem}.txt").read_text().split("\n"):
        p = line.split()
        if len(p) >= 5:
            out.append((names[p[0]], *map(float, p[1:5])))
    return out


def _render(spec, w_px: int, h_px: int, scale: int = 3, corpus: pathlib.Path = RV) -> Image.Image:
    """Render one frame's imagery at `scale` times the on-page size."""
    W, H = w_px * scale, h_px * scale
    base, _ = _series(spec[1], corpus)
    series = str(base.relative_to(corpus)) if base != corpus else ""
    stem = _stem(spec[1], corpus)
    page = Image.open(base / "images" / f"{stem}.jpg").convert("RGB")
    pw, ph = page.size

    if spec[0] == "page":
        overlay = spec[2]
        if overlay:
            d = ImageDraw.Draw(page)
            for nm, xc, yc, bw, bh in _boxes(stem, corpus, series):
                colour = RGB.get(overlay.get(nm), RGB[GREY])
                d.rectangle([(xc - bw / 2) * pw, (yc - bh / 2) * ph,
                             (xc + bw / 2) * pw, (yc + bh / 2) * ph],
                            outline=colour, width=max(3, pw // 150))
        src = page
    else:
        _, _, char, rank = spec
        want = W / H
        def score(bw, bh):
            """Prefer a large box whose shape is close to the frame, so the cover-fit
            below crops little and the character is not cut in half."""
            a = (bw * pw) / (bh * ph)
            return bw * bh * min(a / want, want / a) ** 0.7
        cand = sorted(((score(bw, bh), xc, yc, bw, bh)
                       for nm, xc, yc, bw, bh in _boxes(stem, corpus, series) if nm == char), reverse=True)
        if len(cand) <= rank:
            sys.exit(f"{spec[1]} has no rank-{rank} {char} box")
        _, xc, yc, bw, bh = cand[rank]
        # crop tight to the annotation; the cover-fit below trims to the frame shape
        l, t = (xc - bw / 2) * pw, (yc - bh / 2) * ph
        r, b = (xc + bw / 2) * pw, (yc + bh / 2) * ph
        src = page.crop((int(max(0, l)), int(max(0, t)), int(min(pw, r)), int(min(ph, b))))

    # cover-fit into the frame
    sw, sh = src.size
    k = max(W / sw, H / sh)
    src = src.resize((max(1, int(sw * k)), max(1, int(sh * k))), Image.LANCZOS)
    sw, sh = src.size
    return src.crop(((sw - W) // 2, (sh - H) // 2, (sw - W) // 2 + W, (sh - H) // 2 + H))


def emit_canva(page_id: str, figure: str = "binding") -> None:
    ops = []
    for name, (l, t, w, h, _spec, stroke) in FIGURES[figure].items():
        ops.append({"type": "insert_shape", "page_id": page_id, "left": l, "top": t,
                    "width": w, "height": h, "path": f"M0 0H{w}V{h}H0Z",
                    "view_box_width": w, "view_box_height": h,
                    "color": "#f2f2f2", "stroke_color": stroke,
                    "stroke_weight": 2.4, "corner_rounding": 6})
    print(json.dumps(ops))


def composite(src_pdf: pathlib.Path, dst_pdf: pathlib.Path, figure: str = "binding") -> None:
    import fitz
    doc = fitz.open(src_pdf)
    page = doc[0]
    for name, (l, t, w, h, spec, _stroke) in FIGURES[figure].items():
        img = _render(spec, w, h, corpus=CORPUS.get(figure, RV))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        rect = fitz.Rect(l * PX_TO_PT, t * PX_TO_PT, (l + w) * PX_TO_PT, (t + h) * PX_TO_PT)
        page.insert_image(rect, stream=buf.getvalue(), keep_proportion=False, overlay=True)
        print(f"  {name:8s} {spec[0]:5s} {spec[1]}  -> {rect.width:.1f}x{rect.height:.1f} pt")
    doc.save(dst_pdf)
    print(f"wrote {dst_pdf} ({dst_pdf.stat().st_size} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-canva", metavar="PAGE_ID")
    ap.add_argument("--composite", nargs=2, metavar=("SRC", "DST"))
    ap.add_argument("--figure", default="binding", choices=sorted(FIGURES))
    a = ap.parse_args()
    if a.emit_canva:
        emit_canva(a.emit_canva, a.figure)
    elif a.composite:
        composite(pathlib.Path(a.composite[0]), pathlib.Path(a.composite[1]), a.figure)
    else:
        ap.error("pass --emit-canva or --composite")
