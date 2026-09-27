#!/usr/bin/env python3
"""Build the Re:Verse series directory the evaluation reads from the dataset's own export.

The Re:Verse dataset on the Hugging Face Hub ships its Re:Zero pages as a flat YOLO export,
`data/images/` and `data/labels/`, with fifteen classes: twelve characters and three text-box
classes (dialogue, speech and text boxes). The evaluation reads one series directory in the
repository's common layout, so this keeps the character boxes, drops the text boxes, and writes

    Datasets/Re-Verse/Re-Zero/images/                 symlinks to data/images/
    Datasets/Re-Verse/Re-Zero/annotations/            character boxes only, class ids unchanged
    Datasets/Re-Verse/Re-Zero/category_mapping.json   class id -> character name

    git clone https://huggingface.co/datasets/sochastic/Re-Verse Datasets/Re-Verse
    python scripts/prepare_reverse.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEXT_BOXES = {"dbox", "sbox", "tbox"}


def prepare(src: Path, out: Path) -> int:
    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    keep = {i for i, n in enumerate(names) if n not in TEXT_BOXES}
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    pages = 0
    for label in sorted((src / "data" / "labels").glob("*.txt")):
        lines = [l for l in label.read_text().splitlines() if l.strip() and int(l.split()[0]) in keep]
        (out / "annotations" / label.name).write_text("".join(l + "\n" for l in lines))
        image = src / "data" / "images" / (label.stem + ".jpg")
        link = out / "images" / image.name
        if not link.exists():
            os.symlink(image.resolve(), link)
        pages += 1
    mapping = {str(i): names[i] for i in sorted(keep)}
    (out / "category_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return pages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "Datasets" / "Re-Verse",
                    help="the cloned Re:Verse dataset (holds data.yaml and data/)")
    ap.add_argument("--out", type=Path, default=None, help="default: <src>/Re-Zero")
    args = ap.parse_args(argv)
    out = args.out or args.src / "Re-Zero"
    pages = prepare(args.src, out)
    print(f"[prepare_reverse] {pages} pages -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
