#!/usr/bin/env python3
"""
Convert Manga109 dataset from XML annotations to YOLO format.

Reads the Manga109 zip archive and produces:
    Datasets/manga109/<Title>/images/<page_index>.jpg
    Datasets/manga109/<Title>/annotations/<page_index>.txt   (YOLO format)
    Datasets/manga109/<Title>/category_mapping.json
    configs/training/data_split_manga109.yaml

Usage:
    python scripts/convert_manga109.py
    python scripts/convert_manga109.py --zip-path Datasets/Manga109_released_2023_12_07.zip
    python scripts/convert_manga109.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_ZIP = PROJECT_ROOT / "Datasets" / "Manga109_released_2023_12_07.zip"
OUTPUT_DIR = PROJECT_ROOT / "Datasets" / "manga109"
SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "data_split_manga109.yaml"

# Characters to filter (unnamed/background characters)
FILTER_NAMES = {"other", "others"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Manga109 XML to YOLO format")
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP, help="Path to Manga109 zip")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--split-config", type=Path, default=SPLIT_CONFIG_PATH, help="Output split config")
    parser.add_argument("--train-ratio", type=float, default=0.73, help="Fraction of titles for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    return parser.parse_args()


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, page_width, page_height):
    """Convert pixel coordinates to normalized YOLO format (cx, cy, w, h)."""
    cx = (xmin + xmax) / (2.0 * page_width)
    cy = (ymin + ymax) / (2.0 * page_height)
    w = (xmax - xmin) / page_width
    h = (ymax - ymin) / page_height
    return cx, cy, w, h


def process_title(zf, title, xml_path, img_prefix, output_dir, dry_run):
    """Process a single manga title. Returns (title, num_chars, num_annotations, char_names)."""
    xml_data = zf.read(xml_path)
    root = ET.fromstring(xml_data)

    # Parse characters, filter Other/Others
    characters = root.find("characters")
    char_map = {}  # hex_id -> (class_id, name)
    class_id = 0
    filtered_ids = set()

    for char_elem in characters:
        hex_id = char_elem.attrib["id"]
        name = char_elem.attrib["name"]
        if name.lower() in FILTER_NAMES:
            filtered_ids.add(hex_id)
            continue
        char_map[hex_id] = (class_id, name)
        class_id += 1

    if len(char_map) < 2:
        return title, 0, 0, []

    # Process pages
    pages = root.find("pages")
    total_annotations = 0
    title_dir = output_dir / title
    images_dir = title_dir / "images"
    annotations_dir = title_dir / "annotations"

    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

    for page in pages:
        page_idx = int(page.attrib["index"])
        page_width = int(page.attrib["width"])
        page_height = int(page.attrib["height"])
        page_name = f"{page_idx:03d}"

        # Collect body annotations for this page (skip filtered characters)
        annotations = []
        for body in page.findall("body"):
            char_hex = body.attrib["character"]
            if char_hex in filtered_ids or char_hex not in char_map:
                continue

            cid, _ = char_map[char_hex]
            xmin = int(body.attrib["xmin"])
            ymin = int(body.attrib["ymin"])
            xmax = int(body.attrib["xmax"])
            ymax = int(body.attrib["ymax"])

            # Clamp to page bounds
            xmin = max(0, min(xmin, page_width))
            ymin = max(0, min(ymin, page_height))
            xmax = max(0, min(xmax, page_width))
            ymax = max(0, min(ymax, page_height))

            if xmax <= xmin or ymax <= ymin:
                continue

            cx, cy, w, h = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, page_width, page_height)
            annotations.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not annotations:
            continue

        total_annotations += len(annotations)

        if dry_run:
            continue

        # Write YOLO annotation
        ann_file = annotations_dir / f"{page_name}.txt"
        ann_file.write_text("\n".join(annotations) + "\n")

        # Extract image
        img_zip_path = f"{img_prefix}{title}/{page_name}.jpg"
        try:
            img_data = zf.read(img_zip_path)
            img_file = images_dir / f"{page_name}.jpg"
            img_file.write_bytes(img_data)
        except KeyError:
            # Image might not exist for this page index
            ann_file.unlink(missing_ok=True)
            total_annotations -= len(annotations)

    # Write category mapping
    if not dry_run and total_annotations > 0:
        cat_mapping = {str(cid): name for _, (cid, name) in sorted(char_map.items(), key=lambda x: x[1][0])}
        cat_file = title_dir / "category_mapping.json"
        cat_file.write_text(json.dumps(cat_mapping, indent=2, ensure_ascii=False) + "\n")

    char_names = [name for _, (_, name) in sorted(char_map.items(), key=lambda x: x[1][0])]
    return title, len(char_map), total_annotations, char_names


def generate_split(title_stats, train_ratio, seed):
    """Generate stratified train/val split by character count.

    Sorts titles into bins by character count and splits each bin.
    """
    # Filter to titles with annotations
    valid = [(t, nc, na) for t, nc, na, _ in title_stats if na > 0 and nc >= 2]
    valid.sort(key=lambda x: x[1])  # Sort by num chars

    rng = random.Random(seed)

    # Bin into groups of ~4 for stratification
    bin_size = 4
    train_titles = []
    val_titles = []

    for i in range(0, len(valid), bin_size):
        bin_items = [v[0] for v in valid[i : i + bin_size]]
        rng.shuffle(bin_items)
        n_train = max(1, round(len(bin_items) * train_ratio))
        train_titles.extend(bin_items[:n_train])
        val_titles.extend(bin_items[n_train:])

    train_titles.sort()
    val_titles.sort()
    return train_titles, val_titles


def write_split_config(train_titles, val_titles, output_path):
    """Write YAML split config."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Train/Val split for Manga109 dataset",
        f"# {len(train_titles)} train / {len(val_titles)} val",
        f"# Generated by scripts/convert_manga109.py",
        "",
        "train:",
    ]
    for t in train_titles:
        lines.append(f'  - "{t}"')
    lines.append("")
    lines.append("val:")
    for t in val_titles:
        lines.append(f'  - "{t}"')
    lines.append("")
    output_path.write_text("\n".join(lines))


def main():
    args = parse_args()

    if not args.zip_path.exists():
        print(f"Error: Zip file not found: {args.zip_path}")
        return

    print(f"Opening {args.zip_path}...")
    zf = zipfile.ZipFile(args.zip_path)

    # Find annotation XMLs
    ann_prefix = "Manga109_released_2023_12_07/annotations.v2020.12.18/"
    img_prefix = "Manga109_released_2023_12_07/images/"
    xml_files = sorted([n for n in zf.namelist() if n.startswith(ann_prefix) and n.endswith(".xml")])

    print(f"Found {len(xml_files)} manga titles")
    if args.dry_run:
        print("(dry run: no files will be written)")

    # Process each title
    title_stats = []
    total_chars = 0
    total_anns = 0

    for xml_path in xml_files:
        title = xml_path.split("/")[-1].replace(".xml", "")
        title_name, n_chars, n_anns, char_names = process_title(
            zf, title, xml_path, img_prefix, args.output_dir, args.dry_run,
        )
        title_stats.append((title_name, n_chars, n_anns, char_names))
        total_chars += n_chars
        total_anns += n_anns

        if n_anns > 0:
            print(f"  {title_name}: {n_chars} characters, {n_anns} annotations")
        else:
            print(f"  {title_name}: SKIPPED (no valid annotations)")

    # Summary
    valid_titles = [t for t, nc, na, _ in title_stats if na > 0]
    print(f"\n{'=' * 60}")
    print(f"Conversion complete")
    print(f"  Titles: {len(valid_titles)} / {len(xml_files)}")
    print(f"  Named characters: {total_chars}")
    print(f"  Body annotations: {total_anns}")
    print(f"{'=' * 60}")

    # Generate train/val split
    train_titles, val_titles = generate_split(title_stats, args.train_ratio, args.seed)

    print(f"\nTrain/val split: {len(train_titles)} / {len(val_titles)}")
    print(f"  Train: {train_titles[:5]}...")
    print(f"  Val:   {val_titles[:5]}...")

    if not args.dry_run:
        write_split_config(train_titles, val_titles, args.split_config)
        print(f"\nSplit config written to: {args.split_config}")
        print(f"Dataset written to: {args.output_dir}")

    zf.close()


if __name__ == "__main__":
    main()
