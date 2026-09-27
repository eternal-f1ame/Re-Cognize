"""Why growth pays, or does not, as an exact decomposition rather than an observed pattern.

Across seven corpus-backbone cells the sign of the must-link rule tracks the accuracy of the static gallery: below 35 identity Rank-1 it gains 2.5 to 3.0, at 44 it is flat, at 65 it loses 6.2. Stated that way it is a pattern. It is in fact an identity.

Let G0 be the static gallery and G_q the gallery when query q arrives, so G0 is contained in G_q. Partition the queries by whether growth changed the answer:

    Q_A = { q : nn(q) lies in G_q \\ G0 }       captured by an appended exemplar
    Q_G = the rest

For q in Q_G the nearest entry lies in G0, and because G0 is a subset of G_q it is also the nearest entry *of* G0, so those queries are answered identically with and without growth. Every difference lives in Q_A, and

    delta = c * (p_eff - a_plus)

with c = |Q_A| / |Q| the capture rate, p_eff the share of captured queries whose capturing exemplar carries their own identity, and a_plus the accuracy the *static* gallery achieved on exactly those captured queries. The decomposition is exact, so the measured delta must reproduce it to floating point, which is also the check that this script is right.

a_plus is the term an empirical rule comparing append precision with gallery accuracy leaves out. An appended exemplar is a real crop, so the queries it captures are the ones lying near a real crop of its identity: precisely the queries the gallery already handles best. a_plus therefore exceeds the gallery's overall accuracy a, and growth pays when p_eff > a_plus rather than when p > a. That is why a strong gallery gains nothing from a constraint whose precision comfortably exceeds its average accuracy.

    python analysis/commit_condition.py --device cuda --out results/commit.json
"""
from __future__ import annotations

import argparse
import json
import statistics
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
from recognize.protocol_constants import B_MAX                    # noqa: E402
from recognize.protocols import split_seeds                       # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}


def decompose(features: np.ndarray, labels: np.ndarray, order, k: int, strategy: str, seed: int,
              policy: str, link_groups=None, b_max: Optional[int] = B_MAX) -> Dict:
    """Stream P4 once, recording for every query which gallery it was answered from."""
    seed_map, queries, gallery_only = split_seeds(labels, order, k, strategy, seed)
    seed_idx = sorted([i for v in seed_map.values() for i in v]
                      + [i for v in gallery_only.values() for i in v])
    n0 = len(seed_idx)
    cap = n0 + len(queries)
    g_feat = np.zeros((cap, features.shape[1])); g_lab = np.zeros(cap, dtype=labels.dtype)
    g_src = np.full(cap, -1, dtype=int); active = np.zeros(cap, dtype=bool)
    for r, i in enumerate(seed_idx):
        g_feat[r], g_lab[r], g_src[r], active[r] = features[i], labels[i], i, True
    n_used = n0
    buffers: Dict[int, deque] = {int(c): deque() for c in set(g_lab[:n0])}
    grp_of_row = {}
    if link_groups is not None:
        for r, i in enumerate(seed_idx):
            grp_of_row[r] = link_groups[i]

    captured, cap_right, cap_static_right, n_q = 0, 0, 0, 0
    static_right, grown_right = 0, 0
    for qi in queries:
        idx = np.flatnonzero(active[:n_used])
        sims = g_feat[idx] @ features[qi]
        win = idx[int(np.argmax(sims))]
        pred = int(g_lab[win])
        true = int(labels[qi])
        # what the static gallery alone would have said
        s0 = np.flatnonzero(active[:n0])
        pred0 = int(g_lab[s0[int(np.argmax(g_feat[s0] @ features[qi]))]])
        n_q += 1
        grown_right += int(pred == true); static_right += int(pred0 == true)
        if win >= n0:                                   # answered by an appended exemplar
            captured += 1
            cap_right += int(pred == true)
            cap_static_right += int(pred0 == true)
        # the policy's commitment
        if policy == "frozen":
            continue
        if policy == "oracle":
            target = true
        elif policy == "predicted":
            target = pred
        else:                                            # must-link
            g = link_groups[qi]
            votes: Dict[int, int] = {}
            if g is not None:
                for r in range(n_used):
                    if active[r] and grp_of_row.get(r) == g:
                        votes[int(g_lab[r])] = votes.get(int(g_lab[r]), 0) + 1
            if not votes:
                continue
            target = max(votes.items(), key=lambda kv: kv[1])[0]
        if target not in buffers:
            continue
        buf = buffers[target]
        if b_max is not None and len(buf) >= b_max:
            active[buf.popleft()] = False
        g_feat[n_used], g_lab[n_used], g_src[n_used], active[n_used] = features[qi], target, int(qi), True
        if link_groups is not None:
            grp_of_row[n_used] = link_groups[int(qi)]
        buf.append(n_used); n_used += 1
    return {"n_queries": n_q, "captured": captured,
            "c": captured / max(1, n_q),
            "p_eff": cap_right / max(1, captured),
            "a_plus": cap_static_right / max(1, captured),
            "a": static_right / max(1, n_q),
            "delta": (grown_right - static_right) / max(1, n_q),
            "predicted": (captured / max(1, n_q))
                         * (cap_right / max(1, captured) - cap_static_right / max(1, captured))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["transreid", "magiv2", "magiv3",
                                                       "instructreid", "reid5o"])
    ap.add_argument("--policies", nargs="+", default=["mustlink", "predicted"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", default="random")
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--panels", type=Path, default=Path("results/panels"))
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/commit.json"))
    args = ap.parse_args(argv)

    names = ([n.strip() for n in args.series_file.read_text().split("\n") if n.strip()]
             if args.series_file else load_split()["test"])
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[commit] {ckpt} missing, skipping", flush=True); continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        acc: Dict[str, List[Dict]] = {p: [] for p in args.policies}
        by_series: Dict[str, Dict[str, List[Dict]]] = {}
        for name in names:
            stream = SeriesStream(args.data_root / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            groups = json.loads((args.panels / f"{name.replace(' ', '_')}.json").read_text())["magi_cluster_of_crop"]
            tokens = FT.backbone_tokens(lm.model, stream, tf, batch_size=BATCH[bb], device=args.device)
            for seed in args.seeds:
                sm, _, go = split_seeds(labels, order, args.k, args.strategy, seed)
                g_idx = [i for v in sm.values() for i in v] + [i for v in go.values() for i in v]
                f = np.asarray(FT.extract(lm.model, stream, tf, mode="memory", gallery_idx=g_idx,
                                          batch_size=BATCH[bb], device=args.device,
                                          tokens=tokens).bn, dtype=np.float64)
                for pol in args.policies:
                    r = decompose(f, labels, order, args.k, args.strategy, seed, pol,
                                  link_groups=groups)
                    acc[pol].append(r)
                    # keep the per-run terms so the condition can be tested out of sample:
                    # measure c, p_eff and a+ on one slice of series, predict Delta on another
                    by_series.setdefault(name, {}).setdefault(pol, []).append(
                        {k: r[k] for k in ("c", "p_eff", "a_plus", "a", "delta", "predicted")})
            print(f"[commit] {bb} {name} done", flush=True)
        results[bb] = {p: {k: statistics.fmean(r[k] for r in rs)
                           for k in ("c", "p_eff", "a_plus", "a", "delta", "predicted")}
                       for p, rs in acc.items()}
        results[bb]["by_series"] = by_series

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n{'backbone':<14}{'policy':<11}{'a':>7}{'c':>8}{'p_eff':>8}{'a+':>8}"
          f"{'c(p-a+)':>10}{'measured':>10}{'resid':>8}")
    for bb, per in results.items():
        for pol, m in per.items():
            pred = m["predicted"]          # mean of the per-run identity, not a product of means
            print(f"{bb:<14}{pol:<11}{100*m['a']:>7.1f}{100*m['c']:>7.1f}%{100*m['p_eff']:>8.1f}"
                  f"{100*m['a_plus']:>8.1f}{100*pred:>+10.2f}{100*m['delta']:>+10.2f}"
                  f"{100*(pred-m['delta']):>+8.3f}")
    print(f"\n[commit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
