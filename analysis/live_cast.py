"""Restrict the candidate identities to the ones recently on the page, and rank only against those.

The append policies vary *which exemplars* sit in the gallery for an identity. None varies *which identities are candidates at all*. That is the quantity a reader actually uses: on page 50 you are choosing among the two or three characters in the scene, not among the nine in the volume.

The labels say the restriction is nearly free. Over the 8 test series, a strict backward window of 20 crops contains the query's own identity 97.6 % of the time under temporal seeding while cutting the candidate set from 8.8 identities to 3.0, which moves chance Rank-1 from 11.4 to 33.3. Measured Seq-T Rank-1 is 19.8 to 37.7, nowhere near the 97.6 cap, so the cap costs nothing real.

Why it may survive prediction noise where a rolling *exemplar* window (`rolling_gallery.py`) does not: a live cast is a union over a window, not a decision per crop. An identity stays live if it was named correctly *once* in W crops, and recall of a union is at least the recall of any member. The asymmetry is benign too: wrongly keeping an identity live only means the set shrank less, and only a wrong *exclusion* costs a query.

This measures the restriction alone, on a frozen gallery, so nothing here is confounded with growth. `oracle` fills the window from true labels and is the ceiling; `predicted` fills it from the system's own top-1 on earlier crops and is implementable.

    python analysis/live_cast.py --device cuda --k 5 --strategy temporal \
        --out results/live_t5.json
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
WINDOWS = (5, 10, 20, 40, 80, 160)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def live_cast(features: np.ndarray, labels: np.ndarray, order: List[int],
              seed_map: Dict[int, List[int]], queries: List[int],
              gallery_only: Dict[int, List[int]], *,
              window: Optional[int], source: str, gallery: str) -> Dict:
    """Stream in reading order, ranking each query only against the currently live identities.

    `window` is a count of stream positions, not of queries, so it is wall-clock in reading order. `None` disables the restriction and reproduces the static gallery exactly. A query whose live set is empty falls back to the full gallery, which is what a reader does on the first page.
    """
    entries: Dict[int, List[int]] = {}
    for c, idx in seed_map.items():
        entries.setdefault(int(c), []).extend(idx)
    for c, idx in gallery_only.items():
        entries.setdefault(int(c), []).extend(idx)
    ids = sorted(entries)
    row = {c: r for r, c in enumerate(ids)}

    if gallery == "proto":                       # one L2-normalised mean per identity
        bank = np.stack([_unit(features[entries[c]].mean(axis=0)) for c in ids])
        owner = np.arange(len(ids))
    else:                                        # the bag of exemplars, scored per identity by max
        flat = [(c, i) for c in ids for i in entries[c]]
        bank = features[[i for _, i in flat]]
        owner = np.asarray([row[c] for c, _ in flat])

    qset = set(queries)
    seen: deque = deque(maxlen=window) if window is not None else deque()
    correct, static_ok, alive, n_live, n_fallback = [], [], [], [], 0

    for pos in order:
        if pos not in qset:                      # a gallery crop: its identity is known, so it is live
            seen.append(int(labels[pos]))
            continue
        live = set(seen)
        true = int(labels[pos])
        full = bank @ features[pos]              # the unrestricted ranking, for the decomposition
        s_pred = int(ids[owner[int(np.argmax(full))]])
        if window is None or not live:
            mask = np.ones(len(bank), dtype=bool)
            n_fallback += int(window is not None)
            is_live = True
        else:
            keep = {row[c] for c in live if c in row}
            mask = np.isin(owner, list(keep)) if keep else np.ones(len(bank), dtype=bool)
            if not mask.any():
                mask = np.ones(len(bank), dtype=bool)
                is_live = True
            else:
                is_live = true in live
        pred = int(ids[owner[mask][int(np.argmax(full[mask]))]])
        correct.append(pred == true)
        static_ok.append(s_pred == true)
        alive.append(is_live)
        n_live.append(len(live) if live else len(ids))
        seen.append(true if source == "oracle" else pred)

    c = np.asarray(correct); s = np.asarray(static_ok); L = np.asarray(alive)
    n = max(1, len(c))
    ell = float(L.mean()) if len(L) else 0.0
    # per-run, not a product of means: the identity has to close on this run's own numbers
    on_live = float((c[L].mean() - s[L].mean())) if L.any() else 0.0
    off_live = float(s[~L].mean()) if (~L).any() else 0.0
    return {"R1_identity": float(c.mean()) if len(c) else 0.0,
            "R1_static": float(s.mean()) if len(s) else 0.0,
            "n_queries": len(c),
            "cast_size": float(np.mean(n_live)) if n_live else 0.0,
            "fallback_rate": n_fallback / n,
            # Delta = ell (r+ - a+) - (1 - ell) a-
            "ell": ell, "gain_on_live": on_live, "lost_off_live": off_live,
            "predicted_delta": ell * on_live - (1.0 - ell) * off_live,
            "measured_delta": float(c.mean() - s.mean()) if len(c) else 0.0}


def arms() -> Dict[str, Dict]:
    out = {f"static/{g}": dict(window=None, source="oracle", gallery=g)
           for g in ("exemplar", "proto")}
    for w in WINDOWS:
        for src in ("oracle", "predicted"):
            for g in ("exemplar", "proto"):
                out[f"live{w}/{src}/{g}"] = dict(window=w, source=src, gallery=g)
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
    ap.add_argument("--out", type=Path, default=Path("results/live.json"))
    args = ap.parse_args(argv)

    names = ([n.strip() for n in args.series_file.read_text().split("\n") if n.strip()]
             if args.series_file else load_split()["test"])
    cfgs = arms()
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed{args.train_seed}/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[live] {ckpt} missing, skipping", flush=True)
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
                    cell.setdefault(tag, {})[str(seed)] = live_cast(
                        f, labels, order, seed_map, queries, gallery_only, **kw)
            per_series[name] = cell
            print(f"[live] {bb} {name} done", flush=True)
        results[bb] = per_series
        print(f"[live] {bb} finished", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\n[live] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
