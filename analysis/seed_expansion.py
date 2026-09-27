"""Expand each seed by its own page group, before the stream starts.

Re:Cast does nothing at k=1, and both halves fail for the same reason: with one crop per identity the cast sheet is that crop (a mean of one) and the must-link rule almost never fires, because it needs a seed to share a page group with the query. That is the cell the framework's headline claim is about, so it is the cell worth attacking.

The fix does not need a new signal. A seed sits on a page, and MagiV2's per-page character clustering already says which other crops on that page are the same character, at 89.9 % over the test set. Those crops can join the gallery under the seed's label before the stream starts, with no appearance decision, no threshold and no access to anything the protocol has not already handed over. It is a statement about *seeding*, not about appending: a practitioner who labels one crop gets its page-siblings free, which is how annotation works in practice.

Measured over the test set, expansion adds 1.62 crops per seed at k=1 at 90.6 % precision, taking a one-crop gallery to roughly 2.6 and consuming 2.6 % of the query set. Those consumed crops are removed from the query set for *every* arm, including the baseline, so the comparison is on one query set and the gain cannot come from an easier denominator.

    python analysis/seed_expansion.py --device cuda --k 1 --out results/expand_k1.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402
from recognize.metrics import compute_retrieval_metrics           # noqa: E402
from recognize.protocols import split_seeds                       # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pg", Path(__file__).with_name("prototype_gallery.py"))
pg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pg)

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}


def expand(seed_idx: List[int], labels: np.ndarray, groups) -> Dict[int, int]:
    """Crop -> identity, for the page-group siblings of each seed. The label is the seed's."""
    by_grp: Dict[object, List[int]] = {}
    for i, g in enumerate(groups):
        if g is not None:
            by_grp.setdefault(g, []).append(i)
    out: Dict[int, int] = {}
    seeds = set(seed_idx)
    for i in seed_idx:
        g = groups[i]
        if g is None:
            continue
        for j in by_grp[g]:
            if j not in seeds and j not in out:
                out[int(j)] = int(labels[i])
    return out


def exemplar(features, labels, gallery_idx, gallery_lab, queries) -> Dict:
    """The static exemplar gallery, scored on a given query list."""
    gf, gl = features[gallery_idx], np.asarray(gallery_lab)
    per, correct = [], []
    for qi in queries:
        per.append(compute_retrieval_metrics(features[qi][None], labels[qi][None], gf, gl))
        correct.append(int(gl[int(np.argmax(gf @ features[qi]))]) == int(labels[qi]))
    scored = [m for m in per if m["n_queries"] == 1]
    return {"mAP": float(np.mean([m["mAP"] for m in scored])) if scored else float("nan"),
            "R1_identity": float(np.mean(correct)) if correct else float("nan"),
            "n_queries": len(correct), "n_gallery": len(gallery_idx)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["transreid", "magiv2", "magiv3",
                                                       "instructreid", "reid5o"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--strategy", default="random", choices=("random", "temporal"))
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--panels", type=Path, default=Path("results/panels"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/expand.json"))
    args = ap.parse_args(argv)

    names = load_split()["test"]
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[expand] {ckpt} missing, skipping", flush=True)
            continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in names:
            stream = SeriesStream(Path("Datasets/popcharacters") / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            groups = json.loads((args.panels / f"{name.replace(' ', '_')}.json").read_text())["magi_cluster_of_crop"]
            tokens = FT.backbone_tokens(lm.model, stream, tf, batch_size=BATCH[bb], device=args.device)
            cell: Dict = {}
            for seed in args.seeds:
                seed_map, queries, gallery_only = split_seeds(labels, order, args.k, args.strategy, seed)
                g_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
                f = np.asarray(FT.extract(lm.model, stream, tf, mode="memory", gallery_idx=g_idx,
                                          batch_size=BATCH[bb], device=args.device,
                                          tokens=tokens).bn, dtype=np.float64)
                extra = expand(g_idx, labels, groups)
                q = [i for i in queries if i not in extra]           # one query set for every arm
                wide_idx = g_idx + sorted(extra)
                wide_lab = [int(labels[i]) for i in g_idx] + [extra[i] for i in sorted(extra)]
                base_lab = [int(labels[i]) for i in g_idx]
                cell.setdefault("exemplar/static", {})[str(seed)] = exemplar(f, labels, g_idx, base_lab, q)
                cell.setdefault("exemplar/static+expand", {})[str(seed)] = exemplar(
                    f, labels, wide_idx, wide_lab, q)
                for tag, kw in (("recast", dict(policy="mustlink", window=5, anchor=True)),
                                ("recast+expand", dict(policy="mustlink", window=5, anchor=True))):
                    idx = wide_idx if tag.endswith("expand") else g_idx
                    lab = np.array(wide_lab) if tag.endswith("expand") else labels
                    if tag.endswith("expand"):
                        lab_full = labels.copy()
                        for i, c in extra.items():
                            lab_full[i] = c                          # the label the gallery believes
                        cell.setdefault(tag, {})[str(seed)] = pg.prototypes(
                            f, lab_full, idx, q, link_groups=groups, **kw)
                    else:
                        cell.setdefault(tag, {})[str(seed)] = pg.prototypes(
                            f, labels, idx, q, link_groups=groups, **kw)
                cell.setdefault("expansion", {})[str(seed)] = {
                    "n_added": len(extra), "n_seeds": len(g_idx),
                    "precision": float(np.mean([labels[i] == c for i, c in extra.items()])) if extra else 1.0,
                    "n_queries": len(q)}
            per_series[name] = cell
            print(f"[expand] {bb} {name} done", flush=True)
        results[bb] = per_series

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for bb, ps in results.items():
        def macro(tag, metric="R1_identity"):
            v = [c[tag][s][metric] for c in ps.values() for s in c.get(tag, {})
                 if c[tag][s].get(metric) is not None]
            return 100 * statistics.fmean(v) if v else float("nan")
        base = macro("exemplar/static")
        print(f"\n{bb}  Seq-{'R' if args.strategy=='random' else 'T'} k={args.k}, "
              f"one query set for every arm. static exemplar gallery = {base:.2f}")
        for tag in ("exemplar/static+expand", "recast", "recast+expand"):
            print(f"  {tag:<26}{macro(tag):>8.2f}{macro(tag)-base:>+9.2f}")
        print(f"  expansion adds {macro('expansion','n_added')/max(1e-9,macro('expansion','n_seeds')):.2f} "
              f"crops per seed at {macro('expansion','precision'):.1f}% precision")
    print(f"\n[expand] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
