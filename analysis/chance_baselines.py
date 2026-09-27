"""What a random ranking scores under each protocol, and what the measured numbers look like beside it.

P1 and P2 build their galleries differently. P1's gallery is a fifth of the series, about 98 crops with 14 relevant, while P2 at k=1 holds one seed per identity, about 9 crops with exactly one relevant. A random ranking of nine items where one is right already scores mAP 0.31, because with a single relevant item AP is the reciprocal rank. A ratio of raw mAPs, such as the share of the P1 ceiling that P2 at k=1 recovers, can only be read once both chance floors are known, so this measures them and places every model between chance and a perfect ranking.

Chance is estimated by permutation rather than a formula: the real gallery composition, the real protocol drivers, a random similarity matrix, and the same `compute_retrieval_metrics` that scores the models. (E[AP] under a random ranking is not n_rel/n_gallery: for one relevant item it is H_n/n, which is nearly three times larger at n=9.)

    python analysis/chance_baselines.py [--draws 200]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/report")
sys.path.insert(0, "src")

import tables as T                                          # noqa: E402
from recognize.data import SeriesStream, load_split      # noqa: E402
from recognize.metrics import compute_retrieval_metrics     # noqa: E402
from recognize.protocols import p1_split, split_seeds       # noqa: E402

ROOT = Path("results/popcharacters")
DATA = Path("Datasets/popcharacters")
BACKBONES = ["transreid", "magiv2", "magiv3", "instructreid", "reid5o"]


def chance(labels, q_idx, g_idx, draws: int, rng) -> dict:
    """Mean metrics of a uniformly random ranking of this exact gallery."""
    labels = np.asarray(labels)
    ql, gl = labels[q_idx], labels[g_idx]
    q = np.zeros((len(q_idx), 1)); g = np.zeros((len(g_idx), 1))
    out = {"mAP": [], "R1": []}
    for _ in range(draws):
        m = compute_retrieval_metrics(q, ql, g, gl, sims=rng.normal(size=(len(q_idx), len(g_idx))))
        out["mAP"].append(m["mAP"]); out["R1"].append(m["R1"])
    return {k: float(np.mean(v)) * 100 for k, v in out.items()}


def corrected(measured: float, floor: float) -> float:
    """Where the measurement sits between a random ranking and a perfect one, in percent."""
    return 100.0 * (measured - floor) / (100.0 - floor)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200, help="random rankings per split")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--config", default="memory")
    args = ap.parse_args(argv)

    series = load_split()["test"]
    rng = np.random.default_rng(0)
    floors = {"p1": {"mAP": [], "R1": []}, "p2": {"mAP": [], "R1": []}}
    sizes = {"p1": [], "p2": []}
    for name in series:
        s = SeriesStream(DATA / name)
        labels = np.asarray(s.labels)
        for seed in args.seeds:
            g1, q1, _ = p1_split(labels, seed)
            c1 = chance(labels, q1, g1, args.draws, rng)
            seed_map, q2, gallery_only = split_seeds(labels, s.reading_order, 1, "random", seed)
            g2 = sorted([i for v in seed_map.values() for i in v]
                        + [i for v in gallery_only.values() for i in v])
            c2 = chance(labels, q2, g2, args.draws, rng)
            for proto, c, g in (("p1", c1, g1), ("p2", c2, g2)):
                floors[proto]["mAP"].append(c["mAP"]); floors[proto]["R1"].append(c["R1"])
                sizes[proto].append(len(g))
        print(f"[chance] {name} done", flush=True)

    ch = {p: {m: statistics.fmean(v) for m, v in d.items()} for p, d in floors.items()}
    print(f"\nRandom-ranking baseline, mean over {len(series)} series x {len(args.seeds)} seeds, "
          f"{args.draws} draws each")
    print(f"{'protocol':<12}{'gallery':>9}{'chance mAP':>12}{'chance R1':>11}")
    for p, label in (("p1", "P1"), ("p2", "P2-R@1")):
        print(f"{label:<12}{statistics.fmean(sizes[p]):>9.0f}{ch[p]['mAP']:>12.2f}{ch[p]['R1']:>11.2f}")
    print("The two chance mAPs nearly coincide, but the two chance Rank-1s do not. P2 at k=1 gives")
    print("every query exactly one relevant item out of one seed per identity, so its chance Rank-1")
    print("is 1/identities. P1's gallery is a fifth of every identity's crops, so a query's relevant")
    print("fraction is that identity's share of the series, and the series are imbalanced: the")
    print("query-weighted share is the Simpson index, far above 1/identities.")

    for metric in ("mAP", "R1"):
        print(f"\n{args.config} checkpoints, {metric}, against that baseline")
        print(f"{'backbone':<14}{'P1':>8}{'->skill':>9}{'P2':>9}{'->skill':>9}"
              f"{'P2/P1 raw':>11}{'P2/P1 skill':>13}")
        for bb in BACKBONES:
            tag = f"{bb}_{args.config}_seed0"
            kw = {} if metric == "mAP" else {"metric": "R1"}
            c = T.cell(T.load_series(ROOT, tag, series), "p1", **kw)
            d = T.cell(T.load_series(ROOT, tag, series), "p2", strategy="random", k="1", **kw)
            if not c or not d or c["n_series"] != len(series) or d["n_series"] != len(series):
                print(f"{bb:<14}{'not evaluated on every series':>50}")
                continue
            a, b = c["mean"] * 100, d["mean"] * 100
            sa, sb = corrected(a, ch["p1"][metric]), corrected(b, ch["p2"][metric])
            print(f"{bb:<14}{a:>8.2f}{sa:>9.2f}{b:>9.2f}{sb:>9.2f}"
                  f"{100 * b / a:>10.1f}%{100 * sb / sa:>12.1f}%")
    print("\n'skill' is where the number sits between a random ranking and a perfect one; the raw mAP")
    print("ratio is the share of the P1 ceiling that one seed per identity recovers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
