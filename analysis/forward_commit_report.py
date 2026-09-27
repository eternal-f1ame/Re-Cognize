"""Does pricing an append against the remaining stream buy anything? Reads the output of `forward_commit.py`.

Three questions, and the third is the one that decides whether the mechanism is real or a coincidence of one regime.

1. Under Seq-T, does some phi < 1 beat both the static gallery and append-always? Paired over series x gallery seed, with a sign test, because 40 pairs of a noisy quantity is what we have.
2. Do the per-quartile terms say why? The claim is that p_eff - a_fwd is positive early while p_eff - a_plus is negative overall, so the stationary rule rejects appends the forward rule keeps.
3. The Seq-R control. There a_plus is flat, forward and stationary bars coincide, and the stationary rule is already right, so no phi should beat the static gallery. If an early cutoff helps under Seq-R too, the gain is not about non-stationarity and the explanation is wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ["phi0.0", "phi0.25", "phi0.5", "phi0.75", "phi1.0", "oracle"]
LABEL = {"phi0.0": "static", "phi0.25": "phi=.25", "phi0.5": "phi=.50",
         "phi0.75": "phi=.75", "phi1.0": "append-all", "oracle": "oracle"}


def paired(res, bb, strat, arm, ref):
    """Per (series, seed) difference in identity Rank-1, arm minus ref."""
    d = []
    for _, cell in res[bb].items():
        if strat not in cell:
            continue
        for sd in cell[strat][arm]:
            d.append(100 * (cell[strat][arm][sd]["R1_identity"] - cell[strat][ref][sd]["R1_identity"]))
    return np.array(d)


def signtest(d):
    from math import comb
    n = int((d != 0).sum()); pos = int((d > 0).sum())
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(pos, n + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)
    res = {}
    for f in args.files:
        res.update(json.loads(f.read_text()))

    for strat in ("temporal", "random"):
        name = "Seq-T (chronological)" if strat == "temporal" else "Seq-R (random)  [control]"
        print(f"\n{'='*92}\n{name}: identity Rank-1 against the static gallery, paired over series x seed\n{'='*92}")
        print(f"{'backbone':<14}" + "".join(f"{LABEL[a]:>12s}" for a in ARMS[1:]))
        for bb in res:
            if not any(strat in c for c in res[bb].values()):
                continue
            row = f"{bb:<14}"
            for a in ARMS[1:]:
                d = paired(res, bb, strat, a, "phi0.0")
                star = "*" if signtest(d) < 0.05 else " "
                row += f"{d.mean():>11.2f}{star}"
            print(row)
        print(f"{'':14s}" + "  (* sign test p<0.05 over the 40 pairs)")

        print(f"\n  best gated arm vs append-all (the mechanism's own claim)")
        for bb in res:
            if not any(strat in c for c in res[bb].values()):
                continue
            best, bd = None, None
            for a in ("phi0.25", "phi0.5", "phi0.75"):
                d = paired(res, bb, strat, a, "phi1.0")
                if bd is None or d.mean() > bd.mean():
                    best, bd = a, d
            print(f"    {bb:<14} {LABEL[best]:>8s} - append-all = {bd.mean():+6.2f} "
                  f"(sign p={signtest(bd):.4f}, {int((bd>0).sum())}/{len(bd)} pairs positive)")

    print(f"\n{'='*92}\nPer-quartile terms under Seq-T, append-all, pooled over series/seeds/backbones\n{'='*92}")
    print(f"  {'Q':<3}{'c':>8}{'p_eff':>9}{'a_plus':>9}{'a_fwd':>9}{'p_eff-a_plus':>15}{'p_eff-a_fwd':>14}")
    for q in range(4):
        acc = {kk: [] for kk in ("c", "p_eff", "a_plus", "a_fwd")}
        for bb in res:
            for _, cell in res[bb].items():
                if "temporal" not in cell:
                    continue
                for sd, c in cell["temporal"]["phi1.0"].items():
                    v = c.get("by_quartile", {}).get(str(q))
                    if not v:
                        continue
                    for kk in acc:
                        if v.get(kk) is not None:
                            acc[kk].append(v[kk])
        m = {kk: (np.mean(vv) if vv else float("nan")) for kk, vv in acc.items()}
        print(f"  Q{q+1:<2}{m['c']:>8.3f}{m['p_eff']:>9.3f}{m['a_plus']:>9.3f}{m['a_fwd']:>9.3f}"
              f"{m['p_eff']-m['a_plus']:>15.3f}{m['p_eff']-m['a_fwd']:>14.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
