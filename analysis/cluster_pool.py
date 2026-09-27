"""Name a group of stream crops with one decision, instead of naming every crop on its own.

Seq-T is hard because of where it puts the labels: under temporal seeding 20.5 % of a twenty-crop window is labelled in the opening quarter of the stream and 2.6 to 4.4 % after it, at an identical corpus average (the supervision-geometry figure of `paper/recast_figures.py`). Every anchor studied here is page-local or window-local, so all of them run dry together once the opening pages are past.

If anchors are scarce, one anchor has to serve more crops. There are two ways and only one is open. Chaining an anchor forward crop by crop dies to compounding error, 0.9 per link over twenty-seven pages being 0.06. The other way is to make each anchoring decision cover a group, which means pooling structure the stream supplies without labels.

So: for each query, gather the crops in the causal window that sit within `tau` cosine of it, average the query with them, and match the average against the static gallery. The gallery is never involved in forming the group, nothing is appended, and only crops already read are used.

This is deliberately the idea behind `run_p4`'s `cluster` policy (measured in `mustlink_append.py`) at a different scope. There the group is the query's own MagiV2 page cluster, one to three highly correlated views, so the average is barely a different probe and the policy loses to the static gallery. A causal window of W crops spans pages, so the group is larger and the views are further apart. If scope is what limits the page-cluster version, this works. If the mechanism is wrong, this fails the same way and the two results together close the question.

**The risk.** Pooling trades variance for correlated error. A pure group averages away per-crop noise. An impure group carries one wrong decision to all of its members, and P3 clusters these representations at 59 to 70 % purity. The arms below measure purity and accuracy on pure and impure groups separately so the trade is visible rather than inferred.

`oracle` groups by true label and is the ceiling, never the result.

    python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal \
        --out results/pool_t5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402
from recognize.protocols import split_seeds                       # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
WINDOWS = (10, 20, 40, 80)
TAUS = (0.5, 0.6, 0.7, 0.8)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def cluster_pool(features: np.ndarray, labels: np.ndarray, order: List[int],
                 seed_map: Dict[int, List[int]], queries: List[int],
                 gallery_only: Dict[int, List[int]], *,
                 window: Optional[int], tau: Optional[float], source: str) -> Dict:
    """One pooled decision per group, scored per query against the static gallery.

    `window` is a count of stream positions, so the group is drawn from what has already been read. `tau` is the cosine floor for joining the query's group; `source='oracle'` ignores it and groups by true label instead, which bounds what any grouping rule can be worth here. `window=None` disables pooling and reproduces the per-crop baseline exactly.
    """
    entries: Dict[int, List[int]] = {}
    for c, idx in seed_map.items():
        entries.setdefault(int(c), []).extend(idx)
    for c, idx in gallery_only.items():
        entries.setdefault(int(c), []).extend(idx)
    ids = sorted(entries)
    row = {c: r for r, c in enumerate(ids)}
    flat = [(c, i) for c in ids for i in entries[c]]
    bank = features[[i for _, i in flat]]
    owner = np.asarray([row[c] for c, _ in flat])

    qset = set(queries)
    recent: deque = deque(maxlen=window) if window is not None else deque()
    correct, static_ok, sizes, pure = [], [], [], []

    for pos in order:
        if pos not in qset:
            recent.append(pos)                   # a gallery crop, read like any other
            continue
        true = int(labels[pos])
        s_pred = int(ids[owner[int(np.argmax(bank @ features[pos]))]])
        static_ok.append(s_pred == true)

        if window is None or not recent:
            group = [pos]
        elif source == "oracle":
            group = [pos] + [j for j in recent if int(labels[j]) == true]
        else:
            w = np.asarray(list(recent))
            sims = features[w] @ features[pos]
            group = [pos] + [int(j) for j, s in zip(w, sims) if s >= tau]
        probe = _unit(features[group].mean(axis=0)) if len(group) > 1 else features[pos]
        pred = int(ids[owner[int(np.argmax(bank @ probe))]])
        correct.append(pred == true)
        sizes.append(len(group))
        pure.append(all(int(labels[j]) == true for j in group))
        recent.append(pos)                       # every crop read enters the window, labelled or not

    c = np.asarray(correct); s = np.asarray(static_ok); P = np.asarray(pure)
    out = {"R1_identity": float(c.mean()) if len(c) else 0.0,
           "R1_static": float(s.mean()) if len(s) else 0.0,
           "measured_delta": float(c.mean() - s.mean()) if len(c) else 0.0,
           "n_queries": len(c),
           "group_size": float(np.mean(sizes)) if sizes else 1.0,
           "pure_rate": float(P.mean()) if len(P) else 1.0}
    # the trade the docstring names: what pooling buys on a clean group and costs on a dirty one
    out["delta_on_pure"] = float(c[P].mean() - s[P].mean()) if P.any() else 0.0
    out["delta_on_impure"] = float(c[~P].mean() - s[~P].mean()) if (~P).any() else 0.0
    out["static_on_impure"] = float(s[~P].mean()) if (~P).any() else 0.0
    return out


def arms() -> Dict[str, Dict]:
    out = {"static": dict(window=None, tau=None, source="thresh")}
    for w in WINDOWS:
        out[f"pool{w}/oracle"] = dict(window=w, tau=None, source="oracle")
        for t in TAUS:
            out[f"pool{w}/tau{t}"] = dict(window=w, tau=t, source="thresh")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+",
                    default=["transreid", "magiv2", "magiv3", "instructreid", "reid5o"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", default="temporal", choices=("random", "temporal"))
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/pool.json"))
    args = ap.parse_args(argv)

    names = ([n.strip() for n in args.series_file.read_text().split("\n") if n.strip()]
             if args.series_file else load_split()["test"])
    cfgs = arms()
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed{args.train_seed}/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[pool] {ckpt} missing, skipping", flush=True)
            continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in names:
            stream = SeriesStream(args.data_root / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            tokens = FT.backbone_tokens(lm.model, stream, tf, batch_size=BATCH[bb], device=args.device)
            cell: Dict = {}
            for seed in args.seeds:
                seed_map, queries, gallery_only = split_seeds(labels, order, args.k, args.strategy, seed)
                g_idx = ([i for v in seed_map.values() for i in v]
                         + [i for v in gallery_only.values() for i in v])
                f = np.asarray(FT.extract(lm.model, stream, tf, mode="memory", gallery_idx=g_idx,
                                          batch_size=BATCH[bb], device=args.device,
                                          tokens=tokens).bn, dtype=np.float64)
                for tag, kw in cfgs.items():
                    cell.setdefault(tag, {})[str(seed)] = cluster_pool(
                        f, labels, order, seed_map, queries, gallery_only, **kw)
            per_series[name] = cell
            print(f"[pool] {bb} {name} done", flush=True)
        results[bb] = per_series
        print(f"[pool] {bb} finished", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\n[pool] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
