"""Evaluation data: one held-out series, every crop, reading order.

`SeriesStream` parses a series directory laid out as
``<series>/{images,annotations,category_mapping.json}`` (YOLO boxes, one text
file per page), orders crops by natural page order then annotation index, and
serves padded RGB crops. Every crop of the series is in the stream, not a
subsample, and the order depends only on file names and annotation positions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from .perturb import BOX_KINDS, PIXEL_KINDS, box_perturb, pixel_perturb
from .protocol_constants import CROP_PADDING

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT = REPO_ROOT / "configs" / "training" / "data_split.yaml"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
_NUM = re.compile(r"(\d+)")


def natural_key(text: str) -> tuple:
    """Natural-sort key: 'p2' < 'p10'. Chunks are typed so str and int never compare."""
    return tuple((0, int(c)) if c.isdigit() else (1, c.lower()) for c in _NUM.split(text) if c != "")


def page_order_key(page_name: str, annotation_idx: int) -> tuple:
    """Reading-order key: natural page order, then annotation index.

    'page_2' < 'page_10'; chapter/page numbers embedded in POPCharacters names
    ("X - c001 (web) - p031 [Unknown]") sort as (…, 1, …, 31, …). Filenames are
    never matched against a fixed pattern, so every naming scheme sorts by the same rule.
    """
    return (natural_key(page_name), int(annotation_idx))


@dataclass(frozen=True)
class Crop:
    image_path: Path
    bbox: Tuple[float, float, float, float]  # YOLO: cx, cy, w, h, normalised
    label: int
    category: str
    page_name: str
    annotation_idx: int


class _CropDataset(torch.utils.data.Dataset):
    def __init__(self, stream: "SeriesStream", transform: Optional[Callable]):
        self.stream, self.transform = stream, transform

    def __len__(self) -> int:
        return self.stream.n_crops

    def __getitem__(self, i: int) -> dict:
        img = self.stream.image(i)
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "index": i}


class SeriesStream:
    """All crops of one series in reading order, with padded crops on demand."""

    def __init__(
        self,
        series_dir: Path,
        padding: float = CROP_PADDING,
        exclude_categories: str = "",
        box_noise: Optional[str] = None,
        pixel_noise: Optional[str] = None,
        noise_seed: int = 0,
    ) -> None:
        if box_noise is not None and box_noise not in BOX_KINDS:
            raise ValueError(f"box_noise must be one of {BOX_KINDS}, got {box_noise!r}")
        if pixel_noise is not None and pixel_noise not in PIXEL_KINDS:
            raise ValueError(f"pixel_noise must be one of {PIXEL_KINDS}, got {pixel_noise!r}")
        self.box_noise, self.pixel_noise, self.noise_seed = box_noise, pixel_noise, int(noise_seed)
        self.series_dir = Path(series_dir)
        self.name = self.series_dir.name
        self.padding = float(padding)
        exclude = {c.strip().lower() for c in exclude_categories.split(",") if c.strip()}
        categories = self._load_categories(self.series_dir)
        raw = self._parse(self.series_dir, categories, exclude)
        raw.sort(key=lambda r: page_order_key(r["page_name"], r["annotation_idx"]))
        # labels: contiguous, in first-appearance order along the stream
        remap: Dict[int, int] = {}
        self.identities: Dict[int, str] = {}
        crops: List[Crop] = []
        for r in raw:
            if r["class_id"] not in remap:
                remap[r["class_id"]] = len(remap)
                self.identities[remap[r["class_id"]]] = r["category"]
            crops.append(Crop(r["image_path"], r["bbox"], remap[r["class_id"]], r["category"], r["page_name"], r["annotation_idx"]))
        self.crops = crops
        self.labels = np.array([c.label for c in crops], dtype=np.int64)
        self.reading_order = np.arange(len(crops))
        self._images: Dict[Path, Image.Image] = {}

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def _load_categories(series_dir: Path) -> Dict[int, str]:
        cat_file = series_dir / "category_mapping.json"
        if not cat_file.exists():
            raise FileNotFoundError(f"{cat_file} missing; every series needs an id->name mapping")
        data = json.loads(cat_file.read_text())
        if isinstance(data, dict):
            return {int(k): v for k, v in data.items()}
        return {i: name for i, name in enumerate(data)}

    @staticmethod
    def _parse(series_dir: Path, categories: Dict[int, str], exclude: set) -> List[dict]:
        images_dir = series_dir / "images"
        ann_dir = series_dir / "annotations"
        if not ann_dir.exists():
            ann_dir = series_dir / "labels"
        if not images_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(f"{series_dir} needs images/ and annotations/ (or labels/)")
        rows: List[dict] = []
        for ann in sorted(ann_dir.glob("*.txt")):
            image_path = next((images_dir / f"{ann.stem}{ext}" for ext in _IMAGE_EXTS if (images_dir / f"{ann.stem}{ext}").exists()), None)
            if image_path is None:
                raise FileNotFoundError(f"no image for annotation file {ann}")
            for line_idx, line in enumerate(ann.read_text().splitlines()):
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                name = categories.get(class_id, f"class_{class_id}")
                if name.lower() in exclude:
                    continue
                rows.append({
                    "image_path": image_path, "class_id": class_id, "category": name,
                    "bbox": tuple(float(p) for p in parts[1:5]), "page_name": ann.stem, "annotation_idx": line_idx,
                })
        return rows

    # -- properties -------------------------------------------------------------
    @property
    def n_crops(self) -> int:
        return len(self.crops)

    @property
    def n_identities(self) -> int:
        return len(self.identities)

    # -- crops -------------------------------------------------------------------
    def _page(self, path: Path) -> Image.Image:
        if path not in self._images:
            if len(self._images) > 8:
                self._images.clear()
            self._images[path] = Image.open(path).convert("RGB")
        return self._images[path]

    def _rng(self, i: int) -> np.random.Generator:
        return np.random.default_rng([self.noise_seed, i])

    def _effective_box(self, i: int) -> Tuple[Tuple[float, float, float, float], float]:
        box = self.crops[i].bbox
        if self.box_noise is None:
            return box, 1.0
        return box_perturb(box, self.box_noise, self._rng(i))

    def box_ious(self) -> List[float]:
        """IoU of the box actually cropped with the annotated box, per crop (1.0 without box noise)."""
        return [self._effective_box(i)[1] for i in range(self.n_crops)]

    def image(self, i: int) -> Image.Image:
        """Padded RGB crop i, clipped to the page.

        Same arithmetic as the training crop (`MangaCharacterDataset._crop_bbox`), plus a
        one-pixel minimum size.
        """
        c = self.crops[i]
        page = self._page(c.image_path)
        W, H = page.size
        box, _ = self._effective_box(i)
        cx, cy, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        pad_w, pad_h = w * self.padding, h * self.padding
        x1 = max(0, int(cx - w / 2 - pad_w)); y1 = max(0, int(cy - h / 2 - pad_h))
        x2 = min(W, int(cx + w / 2 + pad_w)); y2 = min(H, int(cy + h / 2 + pad_h))
        crop = page.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
        if self.pixel_noise is not None:
            crop = pixel_perturb(crop, self.pixel_noise, self._rng(i))
        return crop

    def to_dataset(self, transform: Optional[Callable]) -> torch.utils.data.Dataset:
        return _CropDataset(self, transform)


def load_split(path: Path = SPLIT) -> Dict[str, List[str]]:
    """Train / dev / test series lists of a split file; a series may appear in only one list."""
    data = yaml.safe_load(Path(path).read_text())
    split = {k: list(data[k]) for k in ("train", "dev", "test")}
    seen: set = set()
    for k, names in split.items():
        overlap = seen & set(names)
        if overlap:
            raise ValueError(f"series in more than one split: {sorted(overlap)}")
        seen |= set(names)
    return split
