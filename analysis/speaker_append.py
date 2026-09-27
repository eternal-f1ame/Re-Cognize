"""Append the crop a dialogue bubble names, and gate it on precision measured where it is cheap.

The distance analysis (`seed_distance.py`) says the only thing that moves chronological seeding is placing a *correct* reference nearer the query, and the commit condition (`commit_condition.py`) says an append pays while the gallery is worse than the append is precise. The bar is measured per cell (`forward_commit.py`): a+ is 19.7 % on TransReID, 20.2 % on InstructReID and 20.3 % on ReID5o under chronological seeding, against 36.7 % on MagiV2 and 23.1 to 43.7 % under random seeding.

The speaker rule assigns a bubble's named character to the crop the detector associates with that bubble. It averages 16.6 % correct, which is three points under the weakest of those bars and twenty-four under MagiV2's 40.9 % break-even append precision. The average is also the wrong summary: it is 59.5 % on Bakuman, 50.0 % on Dr Stone and 35.1 % on Kagurabachi, all of which clear every Seq-T bar, while Nisekoi supplies 40 % of the firings at 6.9 %.

A rule that is reliable on some volumes and not others is deployable if its precision can be read before it is trusted, which is the same argument the commit condition rests on. Three gates:

    all      append every speaker call
    split    measure precision on a random half of a series' calls, apply to the other half, and
             append only where the measured precision clears that cell's a+. Nothing about the
             scored half enters the decision.
    oracle   gate on the series' true precision: the ceiling the split gate is trying to reach

Appends are causal and scored the way P4 scores them: a crop is ranked against the gallery snapshot that precedes its own update. Mean distance to the nearest own-identity reference is reported alongside, because that is the quantity the mechanism claims to move.

    python analysis/speaker_append.py --device cuda \
        --calls results/names.json --out results/speaker_append.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
B_MAX = 50
# a+ measured per backbone and regime (forward_commit, k=5, 8 series x 5 seeds)
A_PLUS = {"magiv2":  {"temporal": .367, "random": .437},
          "transreid": {"temporal": .197, "random": .231},
          "instructreid": {"temporal": .202, "random": .238},
          "reid5o":   {"temporal": .203, "random": .265},
          "magiv3":   {"temporal": .256, "random": .298}}


def run(feats, labels, order, k, strategy, seed, calls: Dict[int, int],
        *, mode: str, bar: float, rng: Optional[np.random.Generator] = None,
        top1: bool = False, groups: Optional[Sequence] = None) -> Dict:
    """mode in {none, all, split, oracle}. `calls` maps crop -> identity the bubble named.

    With `groups`, a surviving call also names every crop in its page group. The speaker rule is coverage-bound, not precision-bound: 331 calls over 4,058 crops, and the whole measured effect comes from 46 appends on three series. Page grouping is the amplifier: on those three series it turns 92 calls into 320 appends, a 3.48x multiplier, and the group relation itself is 93.9 % precise, so joint precision only falls to 46.2 %, still twice the 19.7 to 25.6 % bar.
    """
    seed_map, queries, gallery_only = split_seeds(labels, order, k, strategy, seed)
    seed_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
    pos = {c: t for t, c in enumerate(order)}
    T = max(1, len(order) - 1)

    use = dict(calls)
    if mode == "none":
        use = {}
    elif mode in ("split", "oracle"):
        items = sorted(calls)
        if mode == "oracle":
            ok = [c for c in items if labels[c] == calls[c]]
            prec = len(ok) / len(items) if items else 0.0
            if prec < bar:
                use = {}
        else:                                   # measure on half, apply to the other half
            idx = np.array(items); rng.shuffle(idx)
            half = idx[: len(idx) // 2]; rest = set(idx[len(idx) // 2:])
            prec = float(np.mean([labels[c] == calls[c] for c in half])) if len(half) else 0.0
            use = {c: calls[c] for c in rest} if prec >= bar else {}

    if groups is not None and use:
        members: Dict[object, List[int]] = defaultdict(list)
        for i, gid in enumerate(groups):
            if gid:
                members[gid].append(i)
        seeds_set = set(seed_idx)
        expanded = dict(use)
        for c, h in list(use.items()):
            gid = groups[c] if c < len(groups) else None
            if not gid:
                continue
            for x in members[gid]:
                if x != c and x not in seeds_set:   # a real system knows its own seeds
                    expanded.setdefault(x, h)
        use = expanded

    cap = len(seed_idx) + len(queries) + len(use) + 4
    g_f = np.zeros((cap, feats.shape[1])); g_id = np.zeros(cap, dtype=np.int64)
    active = np.zeros(cap, dtype=bool); is_seed = np.zeros(cap, dtype=bool)
    n_used = 0
    for i in sorted(seed_idx):
        g_f[n_used], g_id[n_used] = feats[i], labels[i]
        active[n_used] = is_seed[n_used] = True
        n_used += 1
    stat_rows = np.arange(n_used); stat_f, stat_id = g_f[stat_rows].copy(), g_id[stat_rows].copy()
    buffers: Dict[int, deque] = defaultdict(deque)
    # nearest own-identity reference, for the distance the mechanism claims to move
    refs: Dict[int, List[int]] = defaultdict(list)
    for i in sorted(seed_idx):
        refs[int(labels[i])].append(pos[i])

    hit = n = 0
    dsum = 0.0
    qset = set(queries)
    appended = wrong = 0
    for c in order:
        if c in qset:
            true = int(labels[c])
            idx = np.flatnonzero(active[:n_used])
            pred = int(g_id[idx[int(np.argmax(g_f[idx] @ feats[c]))]])
            hit += (pred == true); n += 1
            own = refs.get(true)
            dsum += (min(abs(pos[c] - r) for r in own) / T) if own else 1.0
        if c in use:                                   # the bubble names this crop
            tgt = int(use[c])
            buf = buffers[tgt]
            if len(buf) >= B_MAX:
                active[buf.popleft()] = False
            g_f[n_used], g_id[n_used], active[n_used] = feats[c], tgt, True
            buf.append(n_used); n_used += 1
            appended += 1; wrong += int(labels[c] != tgt)
            refs[tgt].append(pos[c])
        elif top1 and c in qset:
            idx = np.flatnonzero(active[:n_used])
            tgt = int(g_id[idx[int(np.argmax(g_f[idx] @ feats[c]))]])
            buf = buffers[tgt]
            if len(buf) >= B_MAX:
                active[buf.popleft()] = False
            g_f[n_used], g_id[n_used], active[n_used] = feats[c], tgt, True
            buf.append(n_used); n_used += 1
            appended += 1; wrong += int(labels[c] != tgt)
            refs[tgt].append(pos[c])
    return {"R1_identity": hit / n if n else 0.0, "n_queries": n,
            "mean_distance": dsum / n if n else 0.0,
            "n_appended": appended, "wrong_append": wrong / appended if appended else 0.0,
            "n_calls_used": len(use)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+",
                    default=["transreid", "instructreid", "reid5o", "magiv3", "magiv2"])
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--calls", type=Path, default=Path("results/names.json"))
    ap.add_argument("--panels", type=Path, default=Path("results/panels"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--strategies", nargs="+", default=["temporal", "random"])
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/speaker_append.json"))
    args = ap.parse_args(argv)

    raw = {r["series"]: r for r in json.loads(args.calls.read_text())}
    series = [s for s in load_split()["test"] if s in raw]

    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[spkapp] {ckpt} missing, skipping", flush=True); continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in series:
            stream = SeriesStream(args.data_root / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            calls = {int(c): int(h) for c, h, _ in raw[name].get("speaker_calls", [])}
            gf = args.panels / f"{name.replace(' ', '_')}.json"
            groups = json.loads(gf.read_text())["magi_cluster_of_crop"] if gf.exists() else None
            feats = np.asarray(FT.extract(lm.model, stream, tf, mode="none",
                                          batch_size=BATCH[bb], device=args.device).bn,
                               dtype=np.float64)
            cell: Dict = {}
            for strat in args.strategies:
                bar = A_PLUS[bb][strat]
                for sd in args.seeds:
                    rng = np.random.default_rng(1000 + sd)
                    for mode in ("none", "all", "split", "oracle"):
                        cell.setdefault(strat, {}).setdefault(mode, {})[str(sd)] = run(
                            feats, labels, order, args.k, strat, sd, calls,
                            mode=mode, bar=bar, rng=np.random.default_rng(1000 + sd))
                    for mode in ("split", "oracle"):
                        cell[strat].setdefault("linked_" + mode, {})[str(sd)] = run(
                            feats, labels, order, args.k, strat, sd, calls,
                            mode=mode, bar=bar, rng=np.random.default_rng(1000 + sd),
                            groups=groups)
                    cell[strat].setdefault("top1", {})[str(sd)] = run(
                        feats, labels, order, args.k, strat, sd, {},
                        mode="none", bar=bar, top1=True)
            per_series[name] = cell
            print(f"[spkapp] {bb} {name} done ({len(calls)} speaker calls)", flush=True)
        results[bb] = per_series
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print(f"\n[spkapp] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
