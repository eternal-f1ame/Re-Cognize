"""One vector per identity, rolled forward: does averaging make a wrong append survivable?

Two measurements of `rolling_gallery.py` point at the same design. Recency is a large effect (at a matched gallery size of one crop per identity, a recent true crop beats a random seed by 9 mAP and 12 identity Rank-1 on MagiV2), and a *short* exemplar window is worse than a long one under every real policy, because a wrong crop is a larger share of a small gallery and wins top-1 outright.

Averaging inverts that second mechanism. If an identity is one vector and not a bag of exemplars, a wrong crop perturbs the mean by 1/m instead of adding a competing nearest neighbour, so the same window that raises the precision bar for a FIFO lowers it for a prototype. This measures whether that is true, over the same policies, windows and series as the FIFO version in `rolling_gallery.py`.

It also makes the metric honest. Exemplar-level AP with one relevant item per query is 1/rank, so mAP moves with gallery size and the FIFO comparisons are only readable on identity Rank-1. A prototype gallery holds exactly one entry per identity by construction, so its mAP is identity-level and comparable across every row here.

Representations: `mean` keeps the m most recent crops and averages them; `ema` keeps a running average with weight alpha on the newest crop. Both optionally keep the protocol's seeds as a permanent anchor.

    python analysis/prototype_gallery.py --device cuda --k 5 --out results/proto_k5.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402
from recognize.protocol_constants import B_MAX                    # noqa: E402
from recognize.protocols import run_p4, split_seeds               # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
WINDOWS = (1, 2, 3, 5, 10, 20, None)
ALPHAS = (0.1, 0.2, 0.5)
POLICIES = ("frozen", "oracle", "predicted", "mustlink")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def prototypes(features: np.ndarray, labels: np.ndarray, seed_idx: List[int], queries: List[int],
               policy: str, *, window: Optional[int] = None, alpha: Optional[float] = None,
               anchor: bool = True, link_groups: Optional[List] = None) -> Dict:
    """Stream the queries against one vector per identity, updating it as the policy decides.

    `window` averages the crops currently held (the seeds, if anchored, plus the m most recent committed crops); `alpha` instead keeps an exponential average, which has no window at all. A query is scored against the prototypes as they stand before its own update, exactly as P4 scores against the pre-update snapshot.
    """
    ids = sorted({int(labels[i]) for i in seed_idx})
    row = {c: r for r, c in enumerate(ids)}
    d = features.shape[1]
    proto = np.zeros((len(ids), d))
    held: Dict[int, Deque[int]] = {c: deque(maxlen=window) for c in ids}
    anchors: Dict[int, List[int]] = {c: [] for c in ids}
    total = np.zeros((len(ids), d))
    count = np.zeros(len(ids))
    # which identities currently hold a crop of each link group, for the must-link lookup
    votes: Dict[object, Dict[int, int]] = {}

    def bump(grp, c, delta):
        if grp is None:
            return
        v = votes.setdefault(grp, {})
        v[c] = v.get(c, 0) + delta
        if v[c] <= 0:
            v.pop(c)

    def refresh(c):
        r = row[c]
        proto[r] = _unit(total[r] / max(1.0, count[r]))

    def add(c: int, i: int):
        r = row[c]
        if alpha is not None:
            proto[r] = _unit((1 - alpha) * proto[r] + alpha * features[i]) if count[r] else features[i]
            count[r] += 1
            bump(None if link_groups is None else link_groups[i], c, 1)
            return
        old = held[c]
        if window is not None and len(old) == window:
            j = old[0]
            total[r] -= features[j]
            count[r] -= 1
            bump(None if link_groups is None else link_groups[j], c, -1)
        old.append(i)
        total[r] += features[i]
        count[r] += 1
        bump(None if link_groups is None else link_groups[i], c, 1)
        refresh(c)

    for i in sorted(seed_idx):
        c = int(labels[i])
        if anchor:
            anchors[c].append(i)
            r = row[c]
            total[r] += features[i]
            count[r] += 1
            bump(None if link_groups is None else link_groups[i], c, 1)
            if alpha is not None:
                proto[r] = _unit(total[r] / count[r])
            else:
                refresh(c)
        else:
            add(c, i)

    ranks, correct, appended, wrong = [], [], 0, 0
    for qi in queries:
        sims = proto @ features[qi]
        order = np.argsort(-sims)
        true = int(labels[qi])
        if true in row:
            ranks.append(1.0 / (1 + int(np.flatnonzero(order == row[true])[0])))
        correct.append(int(ids[int(order[0])]) == true)
        if policy == "frozen":
            continue
        if policy == "oracle":
            target = true
        elif policy == "predicted":
            target = int(ids[int(order[0])])
        else:
            grp = None if link_groups is None else link_groups[qi]
            cand = votes.get(grp, {}) if grp is not None else {}
            if not cand:
                continue
            target = max(cand.items(), key=lambda kv: kv[1])[0]
        if target not in row:
            continue
        add(target, int(qi))
        appended += 1
        wrong += int(target != true)
    return {"mAP": float(np.mean(ranks)) if ranks else float("nan"),
            "R1_identity": float(np.mean(correct)) if correct else float("nan"),
            "n_queries": len(correct), "n_appended": appended,
            "wrong_append": float(wrong / appended) if appended else 0.0,
            "n_gallery": len(ids)}


def configs():
    out = {}
    for policy in POLICIES:
        for anchor in (True, False):
            a = "+anchor" if anchor else ""
            for w in WINDOWS:
                out[f"proto/{policy}/mean{w if w else 'all'}{a}"] = dict(
                    policy=policy, window=w, anchor=anchor)
            for al in ALPHAS:
                out[f"proto/{policy}/ema{al}{a}"] = dict(policy=policy, alpha=al, anchor=anchor)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["magiv2", "magiv3"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--series", nargs="+", default=None)
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", default="random", choices=("random", "temporal"))
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--panels", type=Path, default=Path("results/panels"))
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/proto.json"))
    args = ap.parse_args(argv)

    names = (args.series or (args.series_file.read_text().split("\n") if args.series_file else None)
             or load_split()["test"])
    names = [n.strip() for n in names if n.strip()]
    cfgs = configs()
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed{args.train_seed}/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[proto] {ckpt} missing, skipping", flush=True)
            continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in names:
            stream = SeriesStream(args.data_root / name)
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
                for tag, kw in cfgs.items():
                    cell.setdefault(tag, {})[str(seed)] = prototypes(
                        f, labels, g_idx, queries, link_groups=groups, **kw)
                for policy in ("frozen", "predicted", "mustlink", "oracle"):
                    extra = {"link_groups": groups} if policy == "mustlink" else {}
                    r = run_p4(f, labels, order, args.k, args.strategy, seed, b_max=B_MAX,
                               update_policy=policy, **extra)
                    cell.setdefault(f"exemplar/{policy}", {})[str(seed)] = {
                        kk: r.get(kk) for kk in ("mAP", "R1_identity", "n_queries",
                                                 "n_appended", "wrong_append")}
            per_series[name] = cell
            print(f"[proto] {bb} {name} done", flush=True)
        results[bb] = per_series

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for bb, ps in results.items():
        def macro(tag, metric="R1_identity"):
            v = [c[tag][s][metric] for c in ps.values() for s in c.get(tag, {})
                 if c[tag][s].get(metric) is not None]
            return 100 * statistics.fmean(v) if v else float("nan")
        base = macro("exemplar/frozen")
        print(f"\n{bb}  Seq-{'R' if args.strategy == 'random' else 'T'} k={args.k}. "
              f"exemplar static gallery = {base:.2f} id-R1")
        print(f"{'gallery':<34}{'id Rank-1':>11}{'vs static':>11}{'mAP':>9}{'correct':>9}")
        for tag in [f"exemplar/{p}" for p in ("frozen", "predicted", "mustlink", "oracle")] + list(cfgs):
            if not any(tag in c for c in ps.values()):
                continue
            v = macro(tag)
            print(f"{tag:<34}{v:>11.2f}{v - base:>+11.2f}{macro(tag, 'mAP'):>9.2f}"
                  f"{100 * (1 - macro(tag, 'wrong_append') / 100):>8.1f}%")
    print(f"\n[proto] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
