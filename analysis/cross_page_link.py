"""Bind crops to crops across pages, which is the relation that can carry an identity forward.

Dialogue anchors bind *names* to crops, and that signal is capped twice over: dialogue carries 0.69 name-bearing bubbles per page, and the only rule that names a specific crop points at the speaker, who is usually not the character being named. Together they yield 55 correct labels over 4,058 crops (`dialogue_names.py`).

The crop-to-crop relation has neither cap. It applies to every crop, and MagiV2's character clustering scores 89.9 to 93.9 per cent on within-page pairs, far above the 19.7 to 25.6 per cent break-even the weak backbones' own galleries set. For TransReID, InstructReID and ReID5o it is an independent signal, not a re-reading of the space their gallery already searches.

Restricted to pairs within a page, though, the relation cannot help. Stream distance is a minimum over the references of an identity, so a link between two crops on the same page moves nothing: page-group propagation triples the appends and moves mean distance only from 0.0869 to 0.0882. A link that spans pages is a different object. It transports an identity from where a seed is to where a query is, which under chronological seeding is the whole problem.

`detect_panels.py` throws that away in two places. It calls the model on `batch_size` pages at a time, default 4, so the clustering never sees more of the volume than that, and it then writes `f"{page}#{label}"`, which forces page scope on whatever cross-page structure survived.

This measures what is there before anything is built on it:

  * must-link precision, split into within-page and cross-page pairs. Cross-page is the harder problem and the number that decides whether the relation is usable at all.
  * the stream distance a cross-page link spans, because a link that reaches 0.01 of the volume buys nothing at a slope of 0.149 per unit.
  * how many distinct (identity, stream position) pairs the relation could add to a chronological gallery, which is the quantity page-group propagation does not move.

    python analysis/cross_page_link.py --device cuda --chunk 4 16 48 \
        --out results/cross_page_link.json
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


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0


def analyse(cluster_of: List[Optional[str]], labels, page_of, pos, T) -> Dict:
    """Split must-link pairs by whether they cross a page, and measure what each kind spans."""
    members: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(cluster_of):
        if c is not None:
            members[c].append(i)
    within = [0, 0]; cross = [0, 0]; spans = []
    ident_pos = set()
    for c, idx in members.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                same = int(labels[i] == labels[j])
                if page_of[i] == page_of[j]:
                    within[0] += same; within[1] += 1
                else:
                    cross[0] += same; cross[1] += 1
                    spans.append(abs(pos[i] - pos[j]) / T)
        for i in idx:
            ident_pos.add((int(labels[i]), int(pos[i] / T * 100)))
    return {"within_ml": within[0] / within[1] if within[1] else None, "within_pairs": within[1],
            "cross_ml": cross[0] / cross[1] if cross[1] else None, "cross_pairs": cross[1],
            "span_mean": float(np.mean(spans)) if spans else 0.0,
            "span_p50": float(np.median(spans)) if spans else 0.0,
            "span_p90": float(np.percentile(spans, 90)) if spans else 0.0,
            "n_clusters": len(members),
            "matched": sum(1 for x in cluster_of if x is not None) / max(1, len(cluster_of)),
            "identity_positions": len(ident_pos)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", nargs="+", type=int, default=[4, 16, 48])
    ap.add_argument("--panel-threshold", type=float, default=0.2)
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/cross_page_link.json"))
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
        order = stream.reading_order
        pos = {c: t for t, c in enumerate(order)}
        T = max(1, len(order) - 1)
        by_page: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(stream.crops):
            by_page[c.page_name].append(i)
        pages = sorted(by_page, key=lambda p: min(pos[i] for i in by_page[p]))
        page_of = [c.page_name for c in stream.crops]

        boxes: Dict[str, List[List[float]]] = {}
        for p, idxs in by_page.items():
            w, h = Image.open(stream.crops[idxs[0]].image_path).size
            boxes[p] = [[(c.bbox[0]-c.bbox[2]/2)*w, (c.bbox[1]-c.bbox[3]/2)*h,
                         (c.bbox[0]+c.bbox[2]/2)*w, (c.bbox[1]+c.bbox[3]/2)*h]
                        for c in (stream.crops[i] for i in idxs)]

        per_chunk: Dict = {}
        for ch in args.chunk:
            cluster_of: List[Optional[str]] = [None] * len(stream.crops)
            try:
                for start in range(0, len(pages), ch):
                    grp = pages[start:start + ch]
                    imgs = [np.array(Image.open(stream.crops[by_page[p][0]].image_path)
                                     .convert("L").convert("RGB")) for p in grp]
                    with torch.no_grad():
                        res = model.predict_detections_and_associations(
                            imgs, panel_detection_threshold=args.panel_threshold)
                    for p, r in zip(grp, res):
                        det = [[float(v) for v in b] for b in r.get("characters", [])]
                        lab = list(r.get("character_cluster_labels", []))
                        for i, cb in zip(by_page[p], boxes[p]):
                            best, bi = 0.0, None
                            for k, db in enumerate(det):
                                v = iou(cb, db)
                                if v > best: best, bi = v, k
                            if bi is not None and best >= args.min_iou and bi < len(lab):
                                # NO page prefix: keep whatever cross-page identity the model gives
                                cluster_of[i] = f"c{start}#{lab[bi]}"
            except torch.cuda.OutOfMemoryError:
                print(f"[xpage] {name} chunk={ch} OOM, skipping", flush=True)
                torch.cuda.empty_cache(); continue
            per_chunk[str(ch)] = analyse(cluster_of, labels, page_of, pos, T)
            r = per_chunk[str(ch)]
            print(f"[xpage] {name} chunk={ch:3d}  within {100*(r['within_ml'] or 0):5.1f}% "
                  f"({r['within_pairs']:5d})  cross {100*(r['cross_ml'] or 0):5.1f}% "
                  f"({r['cross_pairs']:6d})  span p50 {r['span_p50']:.3f}  "
                  f"id-pos {r['identity_positions']:4d}", flush=True)
        out[name] = per_chunk
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\n[xpage] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
