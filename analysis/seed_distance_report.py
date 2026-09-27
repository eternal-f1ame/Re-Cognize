"""Read seed_distance.json and answer one question: does distance explain the Seq-T dip?

Three tests, in increasing order of how much they settle.

1. Accuracy against distance, per regime, on the range where both regimes have queries. If Seq-T sits on Seq-R's curve there, the regimes differ in how distance is *distributed* and not in what distance costs.
2. Within Seq-T alone, the Q1-versus-Q2 gap at matched distance. This is the decisive one, because it needs no overlap between regimes: if the dip survives after conditioning on distance, distance is not what makes Q2 hard.
3. Direct standardisation. Re-weight Seq-T's queries so its distance distribution matches Seq-R's, and report what the regime gap becomes. What remains is the part staleness cannot explain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NBIN = 10


def pool(res, bb=None):
    """Sum the count tables over backbones, series and gallery seeds."""
    out = {}
    for b, per_series in res.items():
        if bb and b != bb:
            continue
        for _, cell in per_series.items():
            for strat, seeds in cell.items():
                for _, c in seeds.items():
                    a = out.setdefault(strat, {"n": np.zeros((NBIN, 4)), "ok": np.zeros((NBIN, 4)),
                                               "n_intro": np.zeros(4), "ok_intro": np.zeros(4),
                                               "n_freq": np.zeros(3), "ok_freq": np.zeros(3)})
                    a["n"] += np.array(c["n"]); a["ok"] += np.array(c["ok"])
                    a["n_intro"] += np.array(c["n_intro"]); a["ok_intro"] += np.array(c["ok_intro"])
                    a["n_freq"] += np.array(c["n_freq"]); a["ok_freq"] += np.array(c["ok_freq"])
    return out


def rate(ok, n):
    return np.divide(ok, n, out=np.full_like(ok, np.nan, dtype=float), where=n > 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--per-backbone", action="store_true")
    args = ap.parse_args(argv)

    res = {}
    for f in args.files:
        res.update(json.loads(f.read_text()))

    names = list(res) if args.per_backbone else [None]
    for bb in names:
        P = pool(res, bb)
        if "random" not in P or "temporal" not in P:
            continue
        R, T = P["random"], P["temporal"]
        tag = bb or "ALL BACKBONES POOLED"
        print(f"\n{'='*78}\n{tag}\n{'='*78}")

        nR, nT = R["n"].sum(1), T["n"].sum(1)
        aR, aT = rate(R["ok"].sum(1), nR), rate(T["ok"].sum(1), nT)
        print("\n1. Static-gallery accuracy against distance to nearest own-identity seed")
        print(f"   {'dist bin':>10s} {'Seq-R n':>9s} {'Seq-R acc':>10s} {'Seq-T n':>9s} {'Seq-T acc':>10s} {'T-R':>7s}")
        both = []
        for b in range(NBIN):
            d = f"{b/NBIN:.1f}-{(b+1)/NBIN:.1f}"
            diff = (aT[b] - aR[b]) if (nR[b] >= 50 and nT[b] >= 50) else np.nan
            if not np.isnan(diff):
                both.append((nR[b] + nT[b], diff))
            print(f"   {d:>10s} {int(nR[b]):9d} {aR[b]:10.3f} {int(nT[b]):9d} {aT[b]:10.3f} "
                  f"{'' if np.isnan(diff) else f'{diff:+7.3f}'}")
        if both:
            w = np.array([x[0] for x in both], float); dd = np.array([x[1] for x in both])
            print(f"   overlap ({len(both)} bins, both n>=50): Seq-T minus Seq-R = {np.average(dd, weights=w):+.4f}")
        print(f"   mean distance: Seq-R {np.average(np.arange(NBIN)/NBIN+0.05, weights=nR):.3f}"
              f"   Seq-T {np.average(np.arange(NBIN)/NBIN+0.05, weights=nT):.3f}")

        print("\n2. Within Seq-T: the Q1-Q2 gap, raw and at matched distance")
        q = rate(T["ok"].sum(0), T["n"].sum(0))
        print(f"   raw by quartile: " + "  ".join(f"Q{i+1} {q[i]:.3f}" for i in range(4))
              + f"   (Q1-Q2 = {q[0]-q[1]:+.3f})")
        num = den = 0.0
        for b in range(NBIN):
            n1, n2 = T["n"][b, 0], T["n"][b, 1]
            if n1 >= 25 and n2 >= 25:
                w = n1 + n2
                num += w * (T["ok"][b, 0] / n1 - T["ok"][b, 1] / n2); den += w
        if den:
            m2 = num / den
            print(f"   distance-matched Q1-Q2 = {m2:+.3f} over {int(den)} queries in shared bins")
            if abs(q[0] - q[1]) > 1e-9:
                note = "reverses" if m2 * (q[0] - q[1]) < 0 else "shrinks to"
                print(f"   => conditioning on distance {note} the gap "
                      f"({q[0]-q[1]:+.3f} -> {m2:+.3f}), so distance accounts for it")
        else:
            print("   no distance bin has >=25 queries in both Q1 and Q2")

        print("\n3. Seq-T standardised to Seq-R's distance distribution")
        wR = nR / max(nR.sum(), 1)
        m = (nT > 0) & (nR > 0)
        if m.any():
            std = np.average(aT[m], weights=wR[m])
            raw_gap = np.nansum(T["ok"]) / max(T["n"].sum(), 1) - np.nansum(R["ok"]) / max(R["n"].sum(), 1)
            print(f"   raw:          Seq-T {np.nansum(T['ok'])/T['n'].sum():.3f} vs Seq-R "
                  f"{np.nansum(R['ok'])/R['n'].sum():.3f}   gap {raw_gap:+.3f}")
            print(f"   standardised: Seq-T {std:.3f} vs Seq-R {np.average(aR[m], weights=wR[m]):.3f}"
                  f"   gap {std - np.average(aR[m], weights=wR[m]):+.3f}")

        print("\n4. The cast-composition alternative")
        ai, af = rate(T["ok_intro"], T["n_intro"]), rate(T["ok_freq"], T["n_freq"])
        aiR, afR = rate(R["ok_intro"], R["n_intro"]), rate(R["ok_freq"], R["n_freq"])
        print("   by identity introduction quartile   Seq-R: " + " ".join(f"{x:.3f}" for x in aiR)
              + "    Seq-T: " + " ".join(f"{x:.3f}" for x in ai))
        print("   by identity frequency tercile       Seq-R: " + " ".join(f"{x:.3f}" for x in afR)
              + "    Seq-T: " + " ".join(f"{x:.3f}" for x in af))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
