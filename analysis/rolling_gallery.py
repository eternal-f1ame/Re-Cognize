"""Is the P4 gallery wrong because it is incorrect, or because it is out of date?

P4 seeds a gallery, grows it with a FIFO of 50 per identity and never evicts a seed, so it accumulates and does not forget. Two separate things could be wrong with it. The exemplars a policy appends are 18-45 % correct, which is a *correctness* problem, and the exemplars that are correct may still be far behind the reader: under temporal seeding the median gap from a query to the nearest seed of its own identity is 26 pages, against 3 under random seeding at k=5. That is a *recency* problem, and the two have different fixes.

This measures recency on its own, by handing the protocol a gallery it could not build: for each identity, the last m crops of that identity to have appeared before the current query, by true label. No policy, no threshold, no errors: the ceiling of a rolling window. Three galleries are compared on P4's own query set and metric:

    seeds only            the static P2 gallery, which is what P4 starts from
    seeds + rolling m     the seeds kept, plus the m most recent true appearances
    rolling m only        no seeds at all, only the m most recent true appearances

and against `run_p4` at the same b_max, which is the same rolling window filled by a real policy. If "rolling m only" with a small m matches or beats the accumulating oracle, recency is worth something over and above correctness and the window should be short. If it does not, correctness is the whole story and the window length is a distraction.

    python analysis/rolling_gallery.py --device cuda --k 5 --out results/rolling_k5.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402
from recognize.metrics import compute_retrieval_metrics           # noqa: E402
from recognize.protocol_constants import B_MAX                    # noqa: E402
from recognize.protocols import run_p4, split_seeds               # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
WINDOWS = (1, 2, 3, 5, 10, 20, 50)
POLICIES = ("frozen", "predicted", "mustlink", "oracle")


def rolling(sims_all: np.ndarray, labels: np.ndarray, seed_idx: List[int], queries: List[int],
            m: int, keep_seeds: bool) -> Dict:
    """Score every query against the m most recent true appearances of each identity before it.

    The window is advanced by the stream, not by a policy: a crop enters its own identity's window the moment it has been delivered, which is the earliest any system could know of it. The seeds are delivered at t = 0, exactly as P4 hands them over, so they start in the window and roll out of it as the character reappears, which is the whole point of the variant. `keep_seeds=True` protects them instead, as P4 does, and measures recency added to the static gallery rather than substituted for it.

    `sims_all` is the full crop-to-crop score matrix, so a step costs a row gather and not a matrix product, and the metric itself is the same `compute_retrieval_metrics` every protocol uses.
    """
    n = len(labels)
    in_gallery = np.zeros(n, dtype=bool)
    protected = np.zeros(n, dtype=bool)
    window: Dict[int, Deque[int]] = {}
    for i in sorted(seed_idx):
        w = window.setdefault(int(labels[i]), deque(maxlen=m))
        if len(w) == m:
            in_gallery[w[0]] = False
        w.append(int(i))
        in_gallery[i] = True
        protected[i] = keep_seeds
    per_query, correct = [], []
    for qi in queries:
        rows = np.flatnonzero(in_gallery | protected)
        if not rows.size:
            continue
        gl = labels[rows]
        s = sims_all[qi, rows]
        per_query.append(compute_retrieval_metrics(
            np.empty((1, 1)), labels[qi][None], np.empty((len(rows), 1)), gl, sims=s[None]))
        correct.append(int(gl[int(np.argmax(s))]) == int(labels[qi]))
        w = window.setdefault(int(labels[qi]), deque(maxlen=m))
        if len(w) == m:
            in_gallery[w[0]] = False
        w.append(int(qi))
        in_gallery[qi] = True
    scored = [x for x in per_query if x["n_queries"] == 1]
    out = {k: float(np.mean([x[k] for x in scored])) if scored else float("nan")
           for k in ("mAP", "R1")}
    out["R1_identity"] = float(np.mean(correct)) if correct else float("nan")
    out["n_queries"] = len(scored)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["magiv2", "magiv3"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--series", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", default="random", choices=("random", "temporal"))
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--panels", type=Path, default=Path("results/panels"))
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/rolling.json"))
    args = ap.parse_args(argv)

    names = (args.series or (args.series_file.read_text().split("\n") if args.series_file else None)
             or load_split()["test"])
    names = [n.strip() for n in names if n.strip()]
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed{args.train_seed}/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[rolling] {ckpt} missing, skipping", flush=True)
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
                f = FT.extract(lm.model, stream, tf, mode="memory", gallery_idx=g_idx,
                               batch_size=BATCH[bb], device=args.device, tokens=tokens).bn
                f = np.asarray(f, dtype=np.float64)
                sims_all = f @ f.T
                for m in WINDOWS:
                    for keep in (True, False):
                        tag = f"rolling{m}" + ("+seeds" if keep else "")
                        cell.setdefault(tag, {})[str(seed)] = rolling(
                            sims_all, labels, g_idx, queries, m, keep)
                for policy in POLICIES:
                    kw = {"link_groups": groups} if policy == "mustlink" else {}
                    for b in WINDOWS + (None,):
                        r = run_p4(f, labels, order, args.k, args.strategy, seed,
                                   b_max=b, update_policy=policy, **kw)
                        tag = f"p4/{policy}/b={b if b is not None else 'inf'}"
                        cell.setdefault(tag, {})[str(seed)] = {
                            kk: r.get(kk) for kk in ("mAP", "R1", "R1_identity", "n_queries")}
            per_series[name] = cell
            print(f"[rolling] {bb} {name} done", flush=True)
        results[bb] = per_series

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for bb, ps in results.items():
        def macro(tag, metric="R1_identity"):
            v = [c[tag][s][metric] for c in ps.values() for s in c.get(tag, {})
                 if c[tag][s][metric] is not None]
            return 100 * statistics.fmean(v) if v else float("nan")
        base = macro("p4/frozen/b=1")
        print(f"\n{bb}  Seq-{'R' if args.strategy == 'random' else 'T'} k={args.k}. "
              f"static gallery = {base:.2f}")
        print(f"{'gallery':<26}{'id Rank-1':>11}{'vs static':>11}{'mAP':>9}")
        rows = ([f"rolling{m}" for m in WINDOWS] + [f"rolling{m}+seeds" for m in WINDOWS]
                + [f"p4/{p}/b={b}" for p in POLICIES for b in list(WINDOWS) + ["inf"]])
        for tag in rows:
            if not any(tag in c for c in ps.values()):
                continue
            v = macro(tag)
            print(f"{tag:<26}{v:>11.2f}{v - base:>+11.2f}{macro(tag, 'mAP'):>9.2f}")
    print(f"\n[rolling] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
