"""Audit the paper's figures for readability at the size they are printed.

For every figure PDF, at its printed width, report:
  * the printed size of every text span (minimum, median, share below the floor),
  * text that collides with other text,
  * text that crosses the edge of a drawn box (a label running out of its chip),
  * text that leaves the canvas,
  * text laid over a placed image.

The printed width is the width at which the paper prints the figure (PRINTED below),
so the sizes are the ones a reader sees on paper, not the ones authored in the source.

    python paper/figure_audit.py              # all figures, table
    python paper/figure_audit.py --render DIR # also write a PNG per figure at print size
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "generated" / "figures"
LINE = 5.5                                   # \linewidth, inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import FLOOR, SCRIPT_FLOOR  # noqa: E402

# the width, in inches, at which the paper prints each figure
PRINTED = {
    "teaser.pdf": LINE,
    "Protocols.pdf": LINE,
    "recast_schematic_a.pdf": LINE,
    "recast_schematic_b.pdf": LINE,
    "recast_binding.pdf": LINE,
    "recast_commit_flow.pdf": LINE,
    "ceiling_recovery.pdf": 0.48 * LINE,
    "p4_closes_gap.pdf": 0.48 * LINE,
    "adaptation.pdf": 0.48 * LINE,
    "one_breakeven.pdf": 0.48 * LINE,
    "supervision_geometry.pdf": 0.48 * LINE,
    "Architecture-Diagram.pdf": LINE,
    "two_pass_open_set_inference_protocol.pdf": LINE,
    "threshold_sensitivity.pdf": LINE,
    "dataset_crops_per_series.pdf": LINE,
    "dataset_boxplot.pdf": LINE,
    "p4_drift.pdf": LINE,                    # appendix figure
}


def _spans(page, s):
    """Text spans with printed size, axis-aligned bbox and true outline.

    A rotated label's axis-aligned bbox is much larger than the label, so two neighbouring
    45-degree tick labels would always 'overlap'. The outline is the rotated rectangle the
    glyphs actually occupy, built from the line direction and the characters' extent.
    """
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            cos, sin = line["dir"]
            for sp in line["spans"]:
                chars = [c for c in sp["chars"] if c["c"].strip()]
                t = "".join(c["c"] for c in sp["chars"]).strip()
                if not t or not chars:
                    continue
                # extent along the baseline direction, measured on char boxes projected onto it
                o = fitz.Point(sp["origin"])
                proj = []
                for c in chars:
                    r = fitz.Rect(c["bbox"])
                    for q in (r.tl, r.tr, r.bl, r.br):
                        proj.append((q.x - o.x) * cos + (q.y - o.y) * sin)
                a, bnd = min(proj), max(proj)
                h = sp["size"] * 0.95
                # the perpendicular that points "up" from the baseline in PDF page space
                px, py = sin, -cos
                p0 = fitz.Point(o.x + cos * a, o.y + sin * a)
                p1 = fitz.Point(o.x + cos * bnd, o.y + sin * bnd)
                poly = [p0 + fitz.Point(-px, -py) * (0.22 * h), p1 + fitz.Point(-px, -py) * (0.22 * h),
                        p1 + fitz.Point(px, py) * (0.78 * h), p0 + fitz.Point(px, py) * (0.78 * h)]
                out.append({"text": t, "size": sp["size"] * s, "bbox": fitz.Rect(sp["bbox"]),
                            "origin": sp["origin"], "rot": abs(sin) > 0.05, "poly": poly})
    return out


def _poly_overlap(pa, pb, shrink=0.12):
    """Separating-axis test on two convex quads, each shrunk toward its centre."""
    def shr(p):
        cx = sum(q.x for q in p) / 4; cy = sum(q.y for q in p) / 4
        return [fitz.Point(cx + (q.x - cx) * (1 - shrink), cy + (q.y - cy) * (1 - shrink)) for q in p]
    A, B = shr(pa), shr(pb)
    for poly in (A, B):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            nx, ny = -e.y, e.x
            pa_ = [q.x * nx + q.y * ny for q in A]
            pb_ = [q.x * nx + q.y * ny for q in B]
            if max(pa_) < min(pb_) or max(pb_) < min(pa_):
                return False
    return True


def _is_script(sp, spans):
    """A sub- or superscript: at most 85 % the size of a span it abuts on the same row.

    Length is no test ("final" in F^final is five letters); abutment plus the size ratio is.
    """
    for o in spans:
        if o is sp or sp["size"] > 0.85 * o["size"]:
            continue
        r = fitz.Rect(o["bbox"]); r.x0 -= 2; r.x1 += 2; r.y0 -= 0.5 * (r.y1 - r.y0); r.y1 += 0.5 * (r.y1 - r.y0)
        if r.intersects(sp["bbox"]):
            return True
    return False


def _shrink(r, f=0.12):
    dx, dy = r.width * f, r.height * f
    return fitz.Rect(r.x0 + dx, r.y0 + dy, r.x1 - dx, r.y1 - dy)


def audit(pdf: Path, width_in: float):
    doc = fitz.open(pdf)
    page = doc[0]
    s = width_in * 72 / page.rect.width                 # printed pt per source pt
    spans = _spans(page, s)
    issues = []
    if not spans:
        return {"name": pdf.name, "spans": 0, "scale": s, "issues": ["no vector text: outlined or raster, check by eye"]}

    body = [sp for sp in spans if not _is_script(sp, spans)]
    scripts = [sp for sp in spans if sp not in body]
    small = [sp for sp in body if sp["size"] < FLOOR - 0.05]
    for sp in small:
        issues.append(f"small {sp['size']:.1f} pt: {sp['text'][:40]!r}")
    for sp in scripts:
        if sp["size"] < SCRIPT_FLOOR - 0.05:
            issues.append(f"small script {sp['size']:.1f} pt: {sp['text'][:20]!r}")

    # text on text
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if a["rot"] or b["rot"]:
                if _poly_overlap(a["poly"], b["poly"]):
                    issues.append(f"overlap: {a['text'][:25]!r} x {b['text'][:25]!r}")
                continue
            ra, rb = _shrink(a["bbox"]), _shrink(b["bbox"])
            if ra.intersects(rb):
                inter = fitz.Rect(ra) & rb
                if inter.get_area() > 0.15 * min(ra.get_area(), rb.get_area()):
                    issues.append(f"overlap: {a['text'][:25]!r} x {b['text'][:25]!r}")

    # text leaving the canvas
    for sp in spans:
        if not page.rect.contains(_shrink(sp["bbox"], 0.05)):
            issues.append(f"off canvas: {sp['text'][:30]!r}")

    # text crossing a drawn box edge: the box must be big enough to be a container
    boxes = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width * s < 14 or r.height * s < 6:
            continue
        if d.get("fill") is None and d.get("color") is None:
            continue
        boxes.append(r)
    for sp in spans:
        t = _shrink(sp["bbox"], 0.08)
        for r in boxes:
            if r.intersects(t) and not r.contains(t):
                # a label sitting just outside the box and touching it is not spillage
                inside = (fitz.Rect(r) & t).get_area() / max(t.get_area(), 1e-6)
                if 0.12 < inside < 0.88:
                    issues.append(f"crosses box edge: {sp['text'][:30]!r}")
                    break

    # text over a placed image
    for im in page.get_image_info():
        r = fitz.Rect(im["bbox"])
        if r.width * s < 10:
            continue
        for sp in spans:
            t = _shrink(sp["bbox"], 0.1)
            if r.intersects(t) and (fitz.Rect(r) & t).get_area() > 0.2 * t.get_area():
                issues.append(f"over an image: {sp['text'][:30]!r}")

    sizes = sorted(sp["size"] for sp in body)
    return {"name": pdf.name, "spans": len(spans), "scale": s, "min": sizes[0],
            "median": statistics.median(sizes), "below": sum(x < FLOOR - 0.05 for x in sizes) / len(sizes),
            "issues": list(dict.fromkeys(issues))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="figure file names; default all in PRINTED")
    ap.add_argument("--render", type=Path, default=None, help="write a PNG of each at print size, 200 dpi")
    ap.add_argument("--quiet", action="store_true", help="counts only")
    ap.add_argument("--figdir", type=Path, default=FIG,
                    help="figure directory (default paper/generated/figures)")
    a = ap.parse_args(argv)
    names = a.names or list(PRINTED)
    bad = 0
    print(f"{'figure':42s} {'print in':>8s} {'min pt':>6s} {'median':>6s} {'<7pt':>5s} issues")
    for n in names:
        p = a.figdir / n
        if not p.exists():
            print(f"{n:42s} missing"); continue
        r = audit(p, PRINTED[n])
        if r["spans"] == 0:
            print(f"{n:42s} {PRINTED[n]:8.2f}  {r['issues'][0]}"); continue
        bad += len(r["issues"])
        print(f"{n:42s} {PRINTED[n]:8.2f} {r['min']:6.1f} {r['median']:6.1f} {r['below']:5.0%} {len(r['issues'])}")
        if not a.quiet:
            for i in r["issues"][:40]:
                print(f"      {i}")
        if a.render:
            a.render.mkdir(parents=True, exist_ok=True)
            doc = fitz.open(p)
            zoom = PRINTED[n] * 72 / doc[0].rect.width * 200 / 72
            doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(a.render / (p.stem + ".png"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
