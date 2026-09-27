"""Price an append against the stream that remains, not the stream so far.

The commit condition Delta = c (p_eff - a_plus) of `commit_condition.py` is exact, but read as a rule for when to append it assumes the static gallery is equally accurate over the whole stream. Under random seeding it is (Seq-R quartiles on MagiV2: 43.6 45.0 46.2 42.1). Under chronological seeding it is not (41.3 36.8 35.2 36.4), and `seed_distance.py` shows why: a query sits 0.094 of the stream from its nearest own-identity seed under Seq-R and 0.407 under Seq-T, and accuracy falls at 0.1493 per unit of that distance.

An append made at time t can only ever answer queries at t' > t, so the accuracy it has to beat is the static gallery's accuracy *over the remainder*, not its accuracy overall. Where a_plus declines, the stationary rule prices a Q1 append against a bar roughly five points too high and rejects appends that pay.

Two things are measured here.

1. The decomposition per quartile: capture rate c, capture precision p_eff, the static gallery's accuracy a_plus on exactly the captured queries, and the forward bar a_fwd, its accuracy on captured queries *after* that quartile. The prediction is that p_eff - a_fwd is positive early while p_eff - a_plus is negative overall.

2. A policy that acts on it: append only while the stream position is below phi, then freeze. If the prediction holds, some phi < 1 beats both the static gallery and append-always under Seq-T, and no phi beats the static gallery under Seq-R, where a_plus is flat and the stationary rule is already right.

Seeds are protected and FIFO eviction at b_max applies to the grown portion, as in `run_p4`.

    python analysis/forward_commit.py --device cuda \
        --backbones magiv2 magiv3 --out results/forward_commit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
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
PHIS = (0.0, 0.25, 0.50, 0.75, 1.0)          # append while position fraction < phi; 0.0 == static
B_MAX = 50


def run(feats: np.ndarray, labels: np.ndarray, order: List[int], k: int, strategy: str, seed: int,
        *, phi: float, oracle: bool = False) -> Dict:
    """One P4 pass with appends gated on stream position. Returns Rank-1 and the decomposition."""
    seed_map, queries, gallery_only = split_seeds(labels, order, k, strategy, seed)
    seed_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
    pos = {c: t for t, c in enumerate(order)}
    T = max(1, len(order) - 1)

    cap = len(seed_idx) + len(queries)
    g_f = np.zeros((cap, feats.shape[1])); g_id = np.zeros(cap, dtype=np.int64)
    g_true = np.zeros(cap, dtype=np.int64); active = np.zeros(cap, dtype=bool)
    is_seed = np.zeros(cap, dtype=bool)
    n_used = 0
    for i in sorted(seed_idx):
        g_f[n_used], g_id[n_used], g_true[n_used] = feats[i], labels[i], labels[i]
        active[n_used] = is_seed[n_used] = True
        n_used += 1
    n_seed_rows = n_used
    buffers: Dict[int, deque] = defaultdict(deque)

    # the static gallery is the seed rows, frozen; used for a_plus and as the control
    stat_rows = np.arange(n_seed_rows)
    stat_f, stat_id = g_f[stat_rows], g_id[stat_rows]

    hit, n = 0, 0
    per_q = []            # (quartile, captured, capture_correct, static_correct)
    for qi in queries:
        true = int(labels[qi])
        idx = np.flatnonzero(active[:n_used])
        sims = g_f[idx] @ feats[qi]
        top = idx[int(np.argmax(sims))]
        pred = int(g_id[top])
        hit += (pred == true); n += 1

        s_top = stat_rows[int(np.argmax(stat_f @ feats[qi]))]
        static_ok = int(stat_id[s_top]) == true
        captured = not is_seed[top]
        # p_eff is the accuracy growth achieves on captured queries, so it turns on the identity
        # the exemplar is FILED under (`g_id`), not on the exemplar's true label. An exemplar that
        # is genuinely of this identity but filed under another still produces a wrong answer.
        per_q.append((min(3, int(pos[qi] / T * 4)), captured,
                      (pred == true) if captured else False, static_ok))

        if pos[qi] / T < phi:
            target = true if oracle else pred
            buf = buffers[target]
            if len(buf) >= B_MAX:
                active[buf.popleft()] = False
            g_f[n_used], g_id[n_used], g_true[n_used] = feats[qi], target, labels[qi]
            active[n_used] = True
            buf.append(n_used); n_used += 1

    arr = np.array(per_q, dtype=float) if per_q else np.zeros((0, 4))
    out = {"R1_identity": hit / n if n else 0.0, "n_queries": n}
    if len(arr):
        cq, capt, cok, sok = arr[:, 0], arr[:, 1].astype(bool), arr[:, 2], arr[:, 3]
        out["R1_static"] = float(sok.mean())
        out["c"] = float(capt.mean())
        out["p_eff"] = float(cok[capt].mean()) if capt.any() else 0.0
        out["a_plus"] = float(sok[capt].mean()) if capt.any() else 0.0
        by_q = {}
        for q in range(4):
            m = cq == q
            mc = m & capt
            later = (cq > q) & capt
            by_q[str(q)] = {
                "c": float(mc.sum() / max(m.sum(), 1)),
                "p_eff": float(cok[mc].mean()) if mc.any() else None,
                "a_plus": float(sok[mc].mean()) if mc.any() else None,
                "a_fwd": float(sok[later].mean()) if later.any() else None,
                "n": int(m.sum()), "n_captured": int(mc.sum()),
            }
        out["by_quartile"] = by_q
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+",
                    default=["magiv2", "magiv3", "transreid", "instructreid", "reid5o"])
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--strategies", nargs="+", default=["temporal", "random"])
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/forward_commit.json"))
    args = ap.parse_args(argv)

    series = ([s.strip() for s in args.series_file.read_text().split("\n") if s.strip()]
              if args.series_file else load_split()["test"])

    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[fwdcommit] {ckpt} missing, skipping", flush=True); continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in series:
            stream = SeriesStream(args.data_root / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            feats = np.asarray(FT.extract(lm.model, stream, tf, mode="none",
                                          batch_size=BATCH[bb], device=args.device).bn,
                               dtype=np.float64)
            cell: Dict = {}
            for strat in args.strategies:
                for sd in args.seeds:
                    for phi in PHIS:
                        cell.setdefault(strat, {}).setdefault(f"phi{phi}", {})[str(sd)] = \
                            run(feats, labels, order, args.k, strat, sd, phi=phi)
                    cell[strat].setdefault("oracle", {})[str(sd)] = \
                        run(feats, labels, order, args.k, strat, sd, phi=1.0, oracle=True)
            per_series[name] = cell
            print(f"[fwdcommit] {bb} {name} done", flush=True)
        results[bb] = per_series
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print(f"\n[fwdcommit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
