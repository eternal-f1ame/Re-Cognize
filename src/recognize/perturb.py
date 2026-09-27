"""Synthetic crop perturbations for the robustness protocol options.

Box noise perturbs the YOLO box before cropping; pixel noise perturbs the crop.
Definitions are fixed here so that every run of a given kind is comparable.
`shiftNN`: each axis is displaced by a fraction drawn uniformly in [0, NN%] of the
box size, with a random sign (mean IoU 0.82 / 0.68 / 0.57 for 10 / 20 / 30%).
`tight07` / `loose13`: width and height scaled by 0.7 / 1.3 about the centre.
`jitterNN`: brightness and contrast multiplied by independent factors in
[1-NN%, 1+NN%]. `blurR`: Gaussian blur of radius R pixels. `occNN`: one grey
rectangle covering NN% of the crop area at a random position.
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BOX_KINDS = ("shift10", "shift20", "shift30", "tight07", "loose13")
PIXEL_KINDS = ("jitter10", "jitter20", "blur2", "blur4", "occ15", "occ30")
Box = Tuple[float, float, float, float]  # YOLO cx, cy, w, h (normalised)


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def box_perturb(bbox: Box, kind: str, rng: np.random.Generator) -> Tuple[Box, float]:
    """Return the perturbed box and its IoU with the original."""
    if kind not in BOX_KINDS:
        raise ValueError(f"unknown box perturbation {kind!r}; choose from {BOX_KINDS}")
    cx, cy, w, h = bbox
    m = re.fullmatch(r"shift(\d+)", kind)
    if m:
        s = int(m.group(1)) / 100.0
        dx = rng.uniform(0.0, s) * w * rng.choice((-1.0, 1.0))
        dy = rng.uniform(0.0, s) * h * rng.choice((-1.0, 1.0))
        new = (cx + dx, cy + dy, w, h)
    elif kind == "tight07":
        new = (cx, cy, 0.7 * w, 0.7 * h)
    elif kind == "loose13":
        new = (cx, cy, 1.3 * w, 1.3 * h)
    else:
        raise ValueError(f"unknown box perturbation {kind!r}; choose from {BOX_KINDS}")
    return new, _iou(bbox, new)


def pixel_perturb(img: Image.Image, kind: str, rng: np.random.Generator) -> Image.Image:
    """Return a perturbed copy of `img` (same size, RGB)."""
    if kind not in PIXEL_KINDS:
        raise ValueError(f"unknown pixel perturbation {kind!r}; choose from {PIXEL_KINDS}")
    m = re.fullmatch(r"jitter(\d+)", kind)
    if m:
        s = int(m.group(1)) / 100.0
        out = ImageEnhance.Brightness(img).enhance(float(rng.uniform(1 - s, 1 + s)))
        return ImageEnhance.Contrast(out).enhance(float(rng.uniform(1 - s, 1 + s)))
    m = re.fullmatch(r"blur(\d+)", kind)
    if m:
        return img.filter(ImageFilter.GaussianBlur(radius=int(m.group(1))))
    m = re.fullmatch(r"occ(\d+)", kind)
    if m:
        frac = int(m.group(1)) / 100.0
        W, H = img.size
        area = frac * W * H
        aspect = float(rng.uniform(0.5, 2.0))
        rw = int(round(min(W, np.sqrt(area * aspect))))
        rh = int(round(min(H, area / max(rw, 1))))
        x0 = int(rng.integers(0, W - rw + 1)); y0 = int(rng.integers(0, H - rh + 1))
        out = img.copy()
        out.paste((128, 128, 128), (x0, y0, x0 + rw, y0 + rh))
        return out
    raise ValueError(f"unknown pixel perturbation {kind!r}; choose from {PIXEL_KINDS}")
