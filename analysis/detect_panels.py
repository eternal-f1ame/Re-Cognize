"""Detect panels with MagiV2 and assign every annotated character crop to one.

The joint-assignment route needs the constraint MagiV2 and FSAC both use: two characters in the same panel are different characters. POPCharacters annotates character boxes only, so the panels have to come from somewhere, and they come free: the MagiV2 checkpoint loaded for its crop encoder also carries a detection transformer.

The output per series is the panel boxes per page, the panel each annotated crop falls in, and MagiV2's own per-page character cluster matched to each annotated crop (`magi_cluster_of_crop`, the link groups the must-link analyses read). It also measures the constraint itself, so that anything built on it rests on a measured precision: how often two crops in the same panel really are different characters. The page-level version of that constraint is useless (two crops on a page are the same identity 52 percent of the time over the 8 test series), so if the panel version is not far better there is nothing to build.

Crops are matched to panels by containment: the panel that covers the largest share of the crop's area. A crop that no panel covers (margin art, a splash spilling outside the grid) is left unassigned and counted.

    python analysis/detect_panels.py --device cuda --out results/panels
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from recognize.backbones import BACKBONE_REGISTRY          # noqa: E402
from recognize.data import SeriesStream, load_split     # noqa: E402


def load_detector(device: str):
    """The full MagiV2 model, detection head included, at the revision the registry pins."""
    from transformers import AutoModel
    spec = BACKBONE_REGISTRY["magiv2"]
    model = AutoModel.from_pretrained(spec.hf_repo, revision=spec.hf_revision, trust_remote_code=True)
    return model.to(device).eval()


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_quality(labels, group_of) -> Dict:
    """Treat a grouping as a constraint source: how often same-group means same identity, and
    different-group (within a page) means different identity."""
    from collections import defaultdict as _dd
    by_page = _dd(list)
    for i, g in enumerate(group_of):
        if g is not None:
            by_page[str(g).split("#")[0]].append(i)
    ml_ok = ml = cl_ok = cl = 0
    for idxs in by_page.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if group_of[i] == group_of[j]:
                    ml += 1; ml_ok += int(labels[i] == labels[j])
                else:
                    cl += 1; cl_ok += int(labels[i] != labels[j])
    return {"must_link_pairs": ml, "must_link_correct": ml_ok / ml if ml else float("nan"),
            "cannot_link_pairs": cl, "cannot_link_correct": cl_ok / cl if cl else float("nan")}


def containment(crop: Sequence[float], panel: Sequence[float]) -> float:
    """Share of the crop's area that lies inside the panel."""
    x1, y1, x2, y2 = crop
    px1, py1, px2, py2 = panel
    iw = max(0.0, min(x2, px2) - max(x1, px1))
    ih = max(0.0, min(y2, py2) - max(y1, py1))
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    return iw * ih / area


def assign(crops: Sequence[Sequence[float]], panels: Sequence[Sequence[float]],
           min_containment: float) -> List[Optional[int]]:
    out = []
    for c in crops:
        if not panels:
            out.append(None); continue
        shares = [containment(c, p) for p in panels]
        best = int(np.argmax(shares))
        out.append(best if shares[best] >= min_containment else None)
    return out


def constraint_purity(labels: Sequence[int], panel_of: Sequence[Optional[int]]) -> Dict:
    """How often two crops sharing a panel are in fact different characters."""
    by_panel = defaultdict(list)
    for lab, pan in zip(labels, panel_of):
        if pan is not None:
            by_panel[pan].append(lab)
    same = diff = 0
    for labs in by_panel.values():
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                if labs[i] == labs[j]:
                    same += 1
                else:
                    diff += 1
    return {"pairs": same + diff, "same_identity": same,
            "cannot_link_correct": diff / (same + diff) if same + diff else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", nargs="+", default=None, help="default: the 8 test series")
    ap.add_argument("--series-file", type=Path, default=None,
                    help="one series name per line; use this instead of --series when a "
                         "name contains a space, which the sbatch wrapper cannot pass")
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--panel-threshold", type=float, default=0.2, help="MagiV2 detection threshold")
    ap.add_argument("--min-containment", type=float, default=0.5)
    ap.add_argument("--min-iou", type=float, default=0.5, help="detection-to-annotation match")
    ap.add_argument("--match", choices=("greedy", "one_to_one"), default="greedy",
                    help="greedy reproduces results/panels; one_to_one is strictly better")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/panels"))
    args = ap.parse_args(argv)

    import torch
    from PIL import Image

    names = (args.series or (args.series_file.read_text().split("\n") if args.series_file else None)
             or load_split()["test"])
    names = [n.strip() for n in names if n.strip()]
    model = load_detector(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []

    for name in names:
        stream = SeriesStream(args.data_root / name)
        by_page: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(stream.crops):
            by_page[c.page_name].append(i)
        pages = sorted(by_page, key=lambda p: min(by_page[p]))
        panels_of_page: Dict[str, List[List[float]]] = {}
        det_of_page: Dict[str, tuple] = {}
        magi_cluster = [None] * len(stream.crops)      # MagiV2's own per-page character cluster

        for start in range(0, len(pages), args.batch_size):
            chunk = pages[start:start + args.batch_size]
            imgs = [np.array(Image.open(stream.crops[by_page[p][0]].image_path).convert("L").convert("RGB"))
                    for p in chunk]
            with torch.no_grad():
                res = model.predict_detections_and_associations(
                    imgs, panel_detection_threshold=args.panel_threshold)
            for p, r in zip(chunk, res):
                panels_of_page[p] = [[float(v) for v in b] for b in r.get("panels", [])]
                det_of_page[p] = ([[float(v) for v in b] for b in r.get("characters", [])],
                                  list(r.get("character_cluster_labels", [])))

        # crop boxes in absolute pixels, page by page
        panel_of = [None] * len(stream.crops)
        for p, idxs in by_page.items():
            panels = panels_of_page.get(p, [])
            w, h = Image.open(stream.crops[idxs[0]].image_path).size
            boxes = []
            for i in idxs:
                cx, cy, bw, bh = stream.crops[i].bbox            # YOLO, normalised
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            for i, a in zip(idxs, assign(boxes, panels, args.min_containment)):
                panel_of[i] = None if a is None else f"{p}#{a}"
            # match MagiV2's detections to our annotations so its clustering can be scored
            det_boxes, det_labels = det_of_page.get(p, ([], []))
            if args.match == "one_to_one" and boxes and det_boxes:
                # Hungarian on IoU. Greedy lets two annotated crops claim one detection and inherit
                # one cluster label, which is a must-link the matching invents rather than one
                # MagiV2 asserts. Removing it is worth +1.4 points of within-page precision at this
                # floor and +4.0 at 0.3 (stage1_refine.py, 8 test series).
                from scipy.optimize import linear_sum_assignment
                M = np.array([[iou(cb, db) for db in det_boxes] for cb in boxes])
                r, c = linear_sum_assignment(-M)
                for bi, k in zip(r, c):
                    if M[bi, k] >= args.min_iou and k < len(det_labels):
                        magi_cluster[idxs[bi]] = f"{p}#{det_labels[k]}"
            else:
                for i, cb in zip(idxs, boxes):
                    best, best_iou = None, 0.0
                    for k, db in enumerate(det_boxes):
                        v = iou(cb, db)
                        if v > best_iou:
                            best, best_iou = k, v
                    if best is not None and best_iou >= args.min_iou and best < len(det_labels):
                        magi_cluster[i] = f"{p}#{det_labels[best]}"

        pur = constraint_purity(stream.labels, panel_of)
        magi = link_quality(stream.labels, magi_cluster)
        matched = sum(1 for x in magi_cluster if x is not None) / max(1, len(magi_cluster))
        n_panels = sum(len(v) for v in panels_of_page.values())
        per_page = statistics.fmean(len(v) for v in panels_of_page.values()) if panels_of_page else 0.0
        unassigned = sum(1 for x in panel_of if x is None) / max(1, len(panel_of))
        (args.out / f"{name.replace(' ', '_')}.json").write_text(json.dumps(
            {"series": name, "panels_per_page": panels_of_page,
             "panel_of_crop": panel_of, "magi_cluster_of_crop": magi_cluster,
             "labels": [int(x) for x in stream.labels],
             "panel_threshold": args.panel_threshold, "min_containment": args.min_containment}, indent=1))
        summary.append((name, len(pages), n_panels, per_page, unassigned, pur, magi, matched))
        print(f"[panels] {name}: {len(pages)} pages, {n_panels} panels "
              f"({per_page:.1f}/page), {100*unassigned:.1f}% crops unassigned, "
              f"same-panel pairs {pur['pairs']}, cannot-link correct {100*pur['cannot_link_correct']:.1f}%; "
              f"magi clusters matched {100*matched:.0f}% of crops, must-link "
              f"{100*magi['must_link_correct']:.1f}% ({magi['must_link_pairs']} pairs), cannot-link "
              f"{100*magi['cannot_link_correct']:.1f}% ({magi['cannot_link_pairs']})", flush=True)

    print(f"\n{'series':<26}{'pg':>5}{'pan/pg':>8}{'panel CL':>10}"
          f"{'magi ML':>9}{'magi CL':>9}{'ML pairs':>10}{'CL pairs':>10}{'matched':>9}")
    for name, npg, npan, per_page, unass, pur, magi, matched in summary:
        print(f"{name[:26]:<26}{npg:>5}{per_page:>8.1f}{100*pur['cannot_link_correct']:>9.1f}%"
              f"{100*magi['must_link_correct']:>8.1f}%{100*magi['cannot_link_correct']:>8.1f}%"
              f"{magi['must_link_pairs']:>10}{magi['cannot_link_pairs']:>10}{100*matched:>8.0f}%")
    tp = sum(s[5]["pairs"] for s in summary); td = sum(s[5]["pairs"] - s[5]["same_identity"] for s in summary)
    ml = sum(s[6]["must_link_pairs"] for s in summary)
    mlok = sum(s[6]["must_link_pairs"] * s[6]["must_link_correct"] for s in summary if s[6]["must_link_pairs"])
    cl = sum(s[6]["cannot_link_pairs"] for s in summary)
    clok = sum(s[6]["cannot_link_pairs"] * s[6]["cannot_link_correct"] for s in summary if s[6]["cannot_link_pairs"])
    print(f"\nConstraint sources across {len(summary)} series, all within-page pairs:")
    print(f"  same page              cannot-link correct  47.8 %   (page-level reference, 8 test series)")
    print(f"  same detected panel    cannot-link correct {100*td/max(1,tp):5.1f} %   ({tp} pairs)")
    print(f"  MagiV2 page clustering must-link  correct {100*mlok/max(1,ml):5.1f} %   ({ml} pairs)")
    print(f"                         cannot-link correct {100*clok/max(1,cl):5.1f} %   ({cl} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
