#!/usr/bin/env python3
"""
Dataset Statistics & Visualization for Re:Cognize Paper

Generates:
1. Statistics tables (POPCharacters + Manga109)
2. Per-series bar chart: crops per series with the character count above each bar, by split
   (POPCharacters: the train/dev/test split of configs/training/data_split.yaml)
3. Character count distribution histogram
4. Crops-per-character distribution (box plot / violin)
5. Long-tail distribution plot
6. Train/val split summary

All outputs are written to paper/generated/figures/ by default. The paper's appendix uses only
figures 2 and 4, printed at \\linewidth, so their type is set at the printed sizes of
paper_style.TYPE; --figures-only writes just those two:

    python paper/dataset_stats.py --figures-only --out paper/generated/figures
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

# Try importing plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42    # TrueType: every glyph stays text
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping plots")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "Datasets"
CONFIGS_DIR = ROOT / "configs" / "training"
OUTPUT_DIR = ROOT / "paper" / "generated" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import TYPE, fig_size, save_fixed, legend_above  # noqa: E402

# Color palette (colorblind-friendly)
TRAIN_COLOR = "#4C72B0"   # blue
VAL_COLOR = "#DD8452"     # orange
ACCENT_COLOR = "#55A868"  # green
INK = "#1b1b1b"


def short_name(name: str) -> str:
    """The series' English title, without the romanised subtitle the directory carries.

    One title is abbreviated: at 7 pt and 45 degrees the longest name sets how much of
    the canvas the axis labels take, and "Assassination Classroom" alone would cost the
    plot a fifth of its height.
    """
    return (name.replace("Hell S Paradise Jigokuraku", "Hell's Paradise")
                .replace("Food Wars Shokugeki No Soma", "Food Wars")
                .replace("Demon Slayer Kimetsu No Yaiba", "Demon Slayer")
                .replace("Kuroko S Basketball", "Kuroko's Basketball")
                .replace("Nisekoi False Love", "Nisekoi")
                .replace("Assassination Classroom", "Assass. Classroom"))


def series_axis(ax, names):
    """23 series names under a full-width axis, at 7 pt.

    At 16 pt per bar no name fits horizontally, so each is set at 45 degrees and
    anchored by its end to its own tick. Adjacent names are 11 pt apart measured
    across the slant, clear of the 7 pt type. (figure_audit.py compares the upright
    bounding boxes of rotated text, which overlap here although the glyphs do not.)
    """
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", va="top", rotation_mode="anchor",
                       fontsize=TYPE["tick"])
    ax.tick_params(axis="x", length=2.0, pad=1.5)
    ax.tick_params(axis="y", labelsize=TYPE["tick"], length=2.0)
    ax.set_xlim(-0.6, len(names) - 0.4)


# POPCharacters uses the split of data_split.yaml: 13 training series, a 2-series
# development split and the 8 test series. Manga109 keeps its conventional train/val volume split.
SPLITS = {"popcharacters": ("train", "dev", "test"), "manga109": ("train", "val")}
SPLIT_COLOR = {"train": TRAIN_COLOR, "dev": ACCENT_COLOR, "test": VAL_COLOR, "val": VAL_COLOR}
SPLIT_LABEL = {"train": "Train", "dev": "Development", "test": "Test", "val": "Val"}


def load_split(dataset: str) -> dict:
    """Load the series split from YAML config."""
    if dataset == "popcharacters":
        path = CONFIGS_DIR / "data_split.yaml"
    else:
        path = CONFIGS_DIR / "data_split_manga109.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def compute_series_stats(series_dir: Path) -> dict:
    """Compute statistics for a single manga series."""
    annot_dir = series_dir / "annotations"
    images_dir = series_dir / "images"
    mapping_file = series_dir / "category_mapping.json"

    # Load category mapping
    if mapping_file.exists():
        with open(mapping_file) as f:
            cat_map = json.load(f)
        num_characters = len(cat_map)
        char_names = {int(k): v for k, v in cat_map.items()}
    else:
        num_characters = 0
        char_names = {}

    # Count pages (images)
    num_pages = 0
    if images_dir.exists():
        num_pages = len([f for f in images_dir.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])

    # Count crops per character from annotations
    crops_per_char = defaultdict(int)
    total_crops = 0
    num_annotations_files = 0

    if annot_dir.exists():
        for annot_file in sorted(annot_dir.glob("*.txt")):
            num_annotations_files += 1
            with open(annot_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        crops_per_char[class_id] += 1
                        total_crops += 1

    # Compute stats. A character counts once it has a crop: the mapping also lists characters
    # that are never annotated, and counting them would give the train split 207 characters
    # against the 198 that the protocols and the training sampler actually see.
    crop_counts = list(crops_per_char.values()) if crops_per_char else [0]
    return {
        "num_characters": len(crops_per_char),
        "num_mapped_characters": num_characters,
        "num_pages": num_pages,
        "total_crops": total_crops,
        "crops_per_char": dict(crops_per_char),
        "char_names": char_names,
        "avg_crops_per_char": np.mean(crop_counts) if total_crops > 0 else 0,
        "median_crops_per_char": np.median(crop_counts) if total_crops > 0 else 0,
        "min_crops_per_char": min(crop_counts) if total_crops > 0 else 0,
        "max_crops_per_char": max(crop_counts) if total_crops > 0 else 0,
        "std_crops_per_char": np.std(crop_counts) if total_crops > 0 else 0,
    }


def compute_dataset_stats(dataset: str) -> dict:
    """Compute full statistics for a dataset."""
    dataset_dir = DATASETS_DIR / dataset
    split = load_split(dataset)

    all_stats = {}
    for split_name in SPLITS[dataset]:
        series_list = split.get(split_name, [])
        split_stats = {}
        for series_name in series_list:
            series_dir = dataset_dir / series_name
            if series_dir.exists():
                stats = compute_series_stats(series_dir)
                split_stats[series_name] = stats
            else:
                print(f"  WARNING: {series_dir} not found, skipping")
        all_stats[split_name] = split_stats

    return all_stats


def print_summary_table(stats: dict, dataset_name: str):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f" {dataset_name} Dataset Statistics")
    print(f"{'='*80}")

    for split_name in stats:
        split_data = stats.get(split_name, {})
        if not split_data:
            continue

        total_chars = sum(s["num_characters"] for s in split_data.values())
        total_crops = sum(s["total_crops"] for s in split_data.values())
        total_pages = sum(s["num_pages"] for s in split_data.values())
        num_series = len(split_data)

        all_crop_counts = []
        for s in split_data.values():
            all_crop_counts.extend(s["crops_per_char"].values())

        avg_crops = np.mean(all_crop_counts) if all_crop_counts else 0
        avg_chars = total_chars / num_series if num_series > 0 else 0

        print(f"\n  {split_name.upper()} Split:")
        print(f"    Series:     {num_series}")
        print(f"    Characters: {total_chars}")
        print(f"    Crops:      {total_crops}")
        print(f"    Pages:      {total_pages}")
        print(f"    Avg crops/char: {avg_crops:.1f}")
        print(f"    Avg chars/series: {avg_chars:.1f}")

        print(f"\n    Per-series breakdown:")
        print(f"    {'Series':<40} {'#Chars':>7} {'#Crops':>7} {'#Pages':>7} {'Avg C/Ch':>9}")
        print(f"    {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
        for name, s in sorted(split_data.items()):
            print(f"    {name:<40} {s['num_characters']:>7} {s['total_crops']:>7} "
                  f"{s['num_pages']:>7} {s['avg_crops_per_char']:>9.1f}")


def generate_latex_table(pop_stats: dict, manga109_stats: dict) -> str:
    """Generate LaTeX table for the paper."""
    rows = []
    for split_name, label in [("train", "Train"), ("val", "Val")]:
        split_data = pop_stats.get(split_name, {})
        n_series = len(split_data)
        n_chars = sum(s["num_characters"] for s in split_data.values())
        n_crops = sum(s["total_crops"] for s in split_data.values())
        all_counts = []
        for s in split_data.values():
            all_counts.extend(s["crops_per_char"].values())
        avg = np.mean(all_counts) if all_counts else 0
        rows.append(f"  & {label} & {n_series} & {n_chars} & {n_crops:,} & {avg:.1f} \\\\")

    # Manga109
    m109_all = {}
    for split_name in ["train", "val"]:
        m109_all.update(manga109_stats.get(split_name, {}))
    n_series_m = len(m109_all)
    n_chars_m = sum(s["num_characters"] for s in m109_all.values())
    n_crops_m = sum(s["total_crops"] for s in m109_all.values())
    all_counts_m = []
    for s in m109_all.values():
        all_counts_m.extend(s["crops_per_char"].values())
    avg_m = np.mean(all_counts_m) if all_counts_m else 0

    latex = f"""% Auto-generated by paper/dataset_stats.py
\\begin{{tabular}}{{ll cccc}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{Split}} & \\textbf{{\\#Series}} & \\textbf{{\\#Characters}} & \\textbf{{\\#Crops}} & \\textbf{{Avg C/Ch}} \\\\
\\midrule
\\multirow{{2}}{{*}}{{POPCharacters}}
{rows[0]}
{rows[1]}
\\midrule
Manga109 & Zero-shot & {n_series_m} & {n_chars_m} & {n_crops_m:,} & {avg_m:.1f} \\\\
\\bottomrule
\\end{{tabular}}"""
    return latex


# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------

def plot_crops_per_series(pop_stats: dict, output_path: Path):
    """Bar chart: total crops per series, colored by train/val split."""
    if not HAS_MPL:
        return

    series_data = []
    for split_name in pop_stats:
        for name, s in pop_stats[split_name].items():
            series_data.append({
                "name": short_name(name),
                "crops": s["total_crops"],
                "chars": s["num_characters"],
                "split": split_name,
                "color": SPLIT_COLOR[split_name],
            })

    # Sort by crop count
    series_data.sort(key=lambda x: x["crops"], reverse=True)

    fig, ax = plt.subplots(figsize=fig_size("full"))
    x = np.arange(len(series_data))
    ax.bar(x, [d["crops"] for d in series_data],
           color=[d["color"] for d in series_data],
           edgecolor="white", linewidth=0.5)

    # Annotate with character count. At 7 pt a "22c" is 13 pt wide on a 16 pt pitch and the
    # labels run together, so the count stands alone and the legend says what it counts.
    for i, d in enumerate(series_data):
        ax.annotate(f'{d["chars"]}', (i, d["crops"]), xytext=(0, 1.5),
                    textcoords="offset points", ha='center', va='bottom',
                    fontsize=TYPE["annot"], color='#555555')

    series_axis(ax, [d["name"] for d in series_data])
    ax.set_ylabel("character crops", fontsize=TYPE["label"])
    ax.set_ylim(0, 1.2 * max(d["crops"] for d in series_data))

    # the short bars on the right leave the top right empty: the legend goes there
    patches = [mpatches.Patch(color=SPLIT_COLOR[k], label=f'{SPLIT_LABEL[k]} ({len(v)})')
               for k, v in pop_stats.items()]
    count_key = mpatches.Patch(color='none', label='above bars: characters')
    ax.legend(handles=patches + [count_key], loc='upper right', ncol=len(patches) + 1,
              fontsize=TYPE["legend"], frameon=False, borderaxespad=0.2,
              handlelength=1.1, handletextpad=0.4, columnspacing=1.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fixed(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_long_tail(pop_stats: dict, output_path: Path):
    """Long-tail distribution: crops per character across all series."""
    if not HAS_MPL:
        return

    all_counts_train = []
    all_counts_val = []
    for split_name in ["train", "val"]:
        for name, s in pop_stats.get(split_name, {}).items():
            counts = sorted(s["crops_per_char"].values(), reverse=True)
            if split_name == "train":
                all_counts_train.extend(counts)
            else:
                all_counts_val.extend(counts)

    all_counts_train.sort(reverse=True)
    all_counts_val.sort(reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram
    all_counts = sorted(all_counts_train + all_counts_val, reverse=True)
    bins = np.arange(0, max(all_counts) + 5, 5)
    ax1.hist(all_counts_train, bins=bins, color=TRAIN_COLOR, alpha=0.7,
             label='Train', edgecolor='white', linewidth=0.3)
    ax1.hist(all_counts_val, bins=bins, color=VAL_COLOR, alpha=0.7,
             label='Val', edgecolor='white', linewidth=0.3)
    ax1.set_xlabel("Crops per Character", fontsize=10)
    ax1.set_ylabel("Number of Characters", fontsize=10)
    ax1.set_title("(a) Crop Count Distribution", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: rank-frequency (long-tail)
    all_combined = sorted(all_counts_train + all_counts_val, reverse=True)
    ax2.bar(range(len(all_combined)), all_combined, width=1.0,
            color=ACCENT_COLOR, edgecolor='none', alpha=0.8)
    ax2.set_xlabel("Character Rank (by crop count)", fontsize=10)
    ax2.set_ylabel("Number of Crops", fontsize=10)
    ax2.set_title("(b) Long-Tail Distribution", fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    save_fixed(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_chars_per_series(pop_stats: dict, output_path: Path):
    """Horizontal bar chart: characters per series, with crops as color intensity."""
    if not HAS_MPL:
        return

    series_data = []
    for split_name, color in [("train", TRAIN_COLOR), ("val", VAL_COLOR)]:
        for name, s in pop_stats.get(split_name, {}).items():
            short_name = (name.replace("Hell S Paradise Jigokuraku", "Hell's Paradise")
                             .replace("Food Wars Shokugeki No Soma", "Food Wars")
                             .replace("Demon Slayer Kimetsu No Yaiba", "Demon Slayer")
                             .replace("Kuroko S Basketball", "Kuroko's Basketball")
                             .replace("Nisekoi False Love", "Nisekoi")
                             .replace("Assassination Classroom", "Assass. Classroom"))
            series_data.append({
                "name": short_name,
                "chars": s["num_characters"],
                "pages": s["num_pages"],
                "crops": s["total_crops"],
                "avg": s["avg_crops_per_char"],
                "split": split_name,
                "color": color,
            })

    # Sort by character count
    series_data.sort(key=lambda x: x["chars"])

    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(series_data))
    bars = ax.barh(y, [d["chars"] for d in series_data],
                   color=[d["color"] for d in series_data],
                   edgecolor="white", linewidth=0.5, height=0.7)

    # Annotate with pages and avg crops
    for i, d in enumerate(series_data):
        ax.text(d["chars"] + 0.3, i, f'{d["pages"]}p, {d["avg"]:.0f}c/ch',
                ha='left', va='center', fontsize=7, color='#555555')

    ax.set_yticks(y)
    ax.set_yticklabels([d["name"] for d in series_data], fontsize=8)
    ax.set_xlabel("Number of Characters", fontsize=10)
    ax.set_title("POPCharacters: Characters per Series", fontsize=12, fontweight='bold')

    train_patch = mpatches.Patch(color=TRAIN_COLOR, label='Train')
    val_patch = mpatches.Patch(color=VAL_COLOR, label='Val')
    ax.legend(handles=[train_patch, val_patch], loc='lower right', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fixed(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_per_series_boxplot(pop_stats: dict, output_path: Path):
    """Box plot: distribution of crops per character within each series."""
    if not HAS_MPL:
        return

    series_data = []
    for split_name in pop_stats:
        for name, s in pop_stats[split_name].items():
            counts = list(s["crops_per_char"].values())
            if counts:
                series_data.append({
                    "name": short_name(name),
                    "counts": counts,
                    "split": split_name,
                    "median": np.median(counts),
                })

    # Sort by median
    series_data.sort(key=lambda x: x["median"])

    fig, ax = plt.subplots(figsize=fig_size("full"))
    positions = range(len(series_data))

    bp = ax.boxplot([d["counts"] for d in series_data],
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))

    for i, (patch, d) in enumerate(zip(bp['boxes'], series_data)):
        patch.set_facecolor(SPLIT_COLOR[d["split"]])
        patch.set_alpha(0.7)

    series_axis(ax, [d["name"] for d in series_data])
    ax.set_ylabel("crops per\ncharacter", fontsize=TYPE["label"], linespacing=1.1)

    patches = [mpatches.Patch(color=SPLIT_COLOR[k], alpha=0.7, label=SPLIT_LABEL[k])
               for k in pop_stats]
    # above the axes: inside, the legend lands on the outliers of the large casts
    legend_above(ax, len(patches), fontsize=TYPE["legend"], handles=patches,
                 labels=[h.get_label() for h in patches])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fixed(fig, output_path)
    print(f"  Saved: {output_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Dataset statistics figures and table.")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR,
                    help="directory to write into (default: paper/generated/figures)")
    ap.add_argument("--figures-only", action="store_true",
                    help="write only the two figures the paper's appendix uses")
    a = ap.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    print("Computing dataset statistics...")

    # POPCharacters
    print("\n[1/2] POPCharacters...")
    pop_stats = compute_dataset_stats("popcharacters")
    print_summary_table(pop_stats, "POPCharacters")

    # Manga109
    print("\n[2/2] Manga109...")
    manga109_stats = compute_dataset_stats("manga109")
    print_summary_table(manga109_stats, "Manga109")

    if not a.figures_only:
        latex = generate_latex_table(pop_stats, manga109_stats)
        latex_path = out / "dataset_stats_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {latex_path}")
        print(latex)

    # Generate plots
    if HAS_MPL:
        print("\nGenerating plots...")
        plot_crops_per_series(pop_stats, out / "dataset_crops_per_series.pdf")
        plot_per_series_boxplot(pop_stats, out / "dataset_boxplot.pdf")
        if not a.figures_only:
            plot_long_tail(pop_stats, out / "dataset_long_tail.pdf")
            plot_chars_per_series(pop_stats, out / "dataset_chars_per_series.pdf")
        print(f"\nAll plots saved to {out}")
    else:
        print("\nSkipping plots (matplotlib not available)")

    # Print combined summary for the paper
    print("\n" + "="*80)
    print(" PAPER-READY SUMMARY")
    print("="*80)
    for ds_name, ds_stats in [("POPCharacters", pop_stats), ("Manga109", manga109_stats)]:
        for split in ds_stats:
            data = ds_stats.get(split, {})
            if not data:
                continue
            n_s = len(data)
            n_c = sum(s["num_characters"] for s in data.values())
            n_cr = sum(s["total_crops"] for s in data.values())
            n_p = sum(s["num_pages"] for s in data.values())
            all_cc = []
            for s in data.values():
                all_cc.extend(s["crops_per_char"].values())
            avg = np.mean(all_cc) if all_cc else 0
            print(f"  {ds_name} {split:>5}: {n_s:>3} series, {n_c:>4} chars, "
                  f"{n_cr:>6} crops, {n_p:>5} pages, {avg:.1f} avg crops/char")


if __name__ == "__main__":
    main()
