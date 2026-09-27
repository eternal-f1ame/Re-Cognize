"""Two ways to attach MagiV2's per-page clustering to annotated crops, and what each costs.

Stage 1 is not precision-limited. Within-page must-link is 91.8 to 96.5 per cent on seven of the eight test series, against a break-even of 19.7 to 25.6. It is coverage-limited: 77 to 92 per cent of crops receive a cluster at all, and a crop with no cluster can never enter a cross-page merge.

Two defects sit in the attachment step rather than in the model.

*Greedy matching with no one-to-one constraint.* `detect_panels.py` gives every annotated crop the detection it overlaps most, and never checks whether that detection is already taken. Two annotated crops can therefore claim one detection and inherit one cluster label, which is a must-link the matching invents rather than one MagiV2 asserts. A one-to-one assignment removes it.

*A fixed IoU floor of 0.5.* A crop below it is dropped entirely, which costs coverage in exactly the place stage 2 needs it.

The panel constraint is not a third option. Two crops in the same panel are different characters only 47.8 to 70.5 per cent of the time, so splitting a cluster on it would be wrong more often than right, which is the same reason the cannot-link half of joint assignment costs 15 to 34 points.

Reports, per configuration: matched coverage, within-page must-link precision, cluster count, and the share of clusters that are singletons (a singleton contributes no within-page link and carries the least evidence into a merge).

    python analysis/stage1_refine.py --device cuda --out results/stage1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402

try:
    from scipy.optimize import linear_sum_assignment
except Exception:                                                  # pragma: no cover
    linear_sum_assignment = None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0


def match_greedy(boxes, dets, floor) -> Dict[int, int]:
    out = {}
    for i, cb in enumerate(boxes):
        best, bi = 0.0, None
        for k, db in enumerate(dets):
            v = iou(cb, db)
            if v > best: best, bi = v, k
        if bi is not None and best >= floor:
            out[i] = bi
    return out


def match_one_to_one(boxes, dets, floor) -> Dict[int, int]:
    """Hungarian on IoU, so no detection is claimed twice."""
    if not boxes or not dets:
        return {}
    M = np.array([[iou(cb, db) for db in dets] for cb in boxes])
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(-M)
        return {int(i): int(j) for i, j in zip(r, c) if M[i, j] >= floor}
    out, used = {}, set()                                   # mutual-best fallback
    for i in np.argsort(-M.max(axis=1)):
        j = int(np.argmax(M[i]))
        if j not in used and M[i, j] >= floor:
            out[int(i)] = j; used.add(j)
    return out


def quality(cluster_of, labels) -> Dict:
    mem = defaultdict(list)
    for i, c in enumerate(cluster_of):
        if c is not None:
            mem[c].append(i)
    ok = n = 0
    for c, idx in mem.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ok += int(labels[idx[a]] == labels[idx[b]]); n += 1
    sing = sum(1 for v in mem.values() if len(v) == 1)
    return {"within_ml": ok / n if n else None, "pairs": n, "clusters": len(mem),
            "singletons": sing, "singleton_share": sing / max(len(mem), 1),
            "matched": sum(1 for x in cluster_of if x is not None) / max(1, len(cluster_of))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ious", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    ap.add_argument("--panel-threshold", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/stage1.json"))
    args = ap.parse_args(argv)

    from transformers import AutoModel
    spec = BACKBONE_REGISTRY["magiv2"]
    model = AutoModel.from_pretrained(spec.hf_repo, revision=spec.hf_revision,
                                      trust_remote_code=True).to(args.device).eval()

    series = ([s.strip() for s in args.series_file.read_text().split("\n") if s.strip()]
              if args.series_file else load_split()["test"])
    out: Dict = {}
    for name in series:
        stream = SeriesStream(args.data_root / name)
        labels = np.asarray(stream.labels)
        by_page: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(stream.crops):
            by_page[c.page_name].append(i)
        pages = sorted(by_page, key=lambda p: min(by_page[p]))

        boxes: Dict[str, List[List[float]]] = {}
        for p, idxs in by_page.items():
            w, h = Image.open(stream.crops[idxs[0]].image_path).size
            boxes[p] = [[(c.bbox[0]-c.bbox[2]/2)*w, (c.bbox[1]-c.bbox[3]/2)*h,
                         (c.bbox[0]+c.bbox[2]/2)*w, (c.bbox[1]+c.bbox[3]/2)*h]
                        for c in (stream.crops[i] for i in idxs)]

        det_of: Dict[str, tuple] = {}
        for start in range(0, len(pages), args.batch_size):
            grp = pages[start:start + args.batch_size]
            imgs = [np.array(Image.open(stream.crops[by_page[p][0]].image_path)
                             .convert("L").convert("RGB")) for p in grp]
            with torch.no_grad():
                res = model.predict_detections_and_associations(
                    imgs, panel_detection_threshold=args.panel_threshold)
            for p, r in zip(grp, res):
                det_of[p] = ([[float(v) for v in b] for b in r.get("characters", [])],
                             list(r.get("character_cluster_labels", [])))

        cell: Dict = {}
        for how, fn in (("greedy", match_greedy), ("one_to_one", match_one_to_one)):
            for fl in args.ious:
                cluster_of: List[Optional[str]] = [None] * len(stream.crops)
                for p, idxs in by_page.items():
                    dets, labs = det_of.get(p, ([], []))
                    for bi, k in fn(boxes[p], dets, fl).items():
                        if k < len(labs):
                            cluster_of[idxs[bi]] = f"{p}#{labs[k]}"
                cell[f"{how}@{fl}"] = quality(cluster_of, labels)
        out[name] = cell
        b = cell["one_to_one@0.5"]; g = cell["greedy@0.5"]
        print(f"[stage1] {name}: greedy .5 {100*g['matched']:.0f}%/{100*(g['within_ml'] or 0):.1f}%  "
              f"1-1 .5 {100*b['matched']:.0f}%/{100*(b['within_ml'] or 0):.1f}%", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\n[stage1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
