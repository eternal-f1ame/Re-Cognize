"""Is chronological seeding a staleness problem, or a cast-composition problem?

Under Seq-T the static gallery is U-shaped over the stream and under Seq-R it is flat. Measured at k=5 over 8 series and 5 gallery seeds, identity Rank-1 by quartile:

    MagiV2   Seq-R  43.6 45.0 46.2 42.1     Seq-T  41.3 36.8 35.2 36.4
    MagiV3   Seq-R  30.3 30.5 32.8 29.2     Seq-T  30.0 24.4 24.6 27.1

Seq-T falls from Q1 to Q2 on five of five backbones, Seq-R on one, and the oracle headroom under Seq-T peaks in Q2 on five of five. So the damage and the recoverable headroom sit in the same place.

Two mechanisms produce that shape and they imply different remedies.

*Staleness.* Chronological seeds sit at each identity's introduction, so a query drifts away from its own references as the stream runs. Seq-R spreads seeds through the stream, so its queries always have a temporally near reference. If this is the cause, then accuracy is a function of distance-to-nearest own-seed and the two regimes lie on *one* curve, separated only by how that distance is distributed. The remedy is to carry early evidence forward, and the commit condition has to be priced against the accuracy of the stream that *remains*, not the stream so far.

*Cast composition.* Identities introduced mid-stream spend their first k appearances as seeds and have few crops left to query, so the Q2-Q3 dip would be a statement about which identities are being asked about rather than about distance. If this is the cause, the curves separate at matched distance and carrying evidence forward cannot help.

This measures the discriminator: static-gallery accuracy against distance to the nearest own-identity seed, cross-tabulated with stream quartile, identity introduction quartile and identity frequency, under both regimes. No memory, no growth: the frozen gallery only, which is the thing whose shape is in question.

    python analysis/seed_distance.py --device cuda \
        --backbones magiv2 magiv3 --out results/seed_distance.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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
NBIN = 10                      # bins of normalised distance-to-nearest-own-seed, over [0, 1]


def run_cell(feats: np.ndarray, labels: np.ndarray, order: List[int],
             seed_map: Dict[int, List[int]], queries: List[int],
             gallery_only: Dict[int, List[int]]) -> Dict:
    """Score every query against the frozen gallery, tagged by distance and by position."""
    pos = {c: t for t, c in enumerate(order)}          # reading-order index of each crop
    T = max(1, len(order) - 1)

    entries: Dict[int, List[int]] = defaultdict(list)
    for src in (seed_map, gallery_only):
        for c, idx in src.items():
            entries[int(c)].extend(idx)
    ids = sorted(entries)
    flat = [(c, i) for c in ids for i in entries[c]]
    bank = feats[[i for _, i in flat]]
    owner = np.asarray([c for c, _ in flat])

    first_seen: Dict[int, int] = {}
    for t, c in enumerate(order):
        first_seen.setdefault(int(labels[c]), t)
    freq = defaultdict(int)
    for c in labels:
        freq[int(c)] += 1

    # cells: [distance bin][query quartile] -> (n, n_correct); plus per-identity-age tables
    n = np.zeros((NBIN, 4)); ok = np.zeros((NBIN, 4))
    n_intro = np.zeros(4); ok_intro = np.zeros(4)           # by identity introduction quartile
    n_freq = np.zeros(3); ok_freq = np.zeros(3)             # by identity frequency tercile
    dsum = 0.0; dn = 0

    fq = sorted(freq.values())
    t1 = fq[len(fq) // 3] if fq else 0
    t2 = fq[2 * len(fq) // 3] if fq else 0

    for q in queries:
        true = int(labels[q])
        own = seed_map.get(true, [])
        if not own:
            continue
        d = min(abs(pos[q] - pos[s]) for s in own) / T     # normalised stream distance
        pred = int(owner[int(np.argmax(bank @ feats[q]))])
        hit = float(pred == true)

        b = min(NBIN - 1, int(d * NBIN))
        qt = min(3, int(pos[q] / T * 4))
        n[b, qt] += 1; ok[b, qt] += hit
        iq = min(3, int(first_seen[true] / T * 4))
        n_intro[iq] += 1; ok_intro[iq] += hit
        ft = 0 if freq[true] <= t1 else (1 if freq[true] <= t2 else 2)
        n_freq[ft] += 1; ok_freq[ft] += hit
        dsum += d; dn += 1

    return {"n": n.tolist(), "ok": ok.tolist(),
            "n_intro": n_intro.tolist(), "ok_intro": ok_intro.tolist(),
            "n_freq": n_freq.tolist(), "ok_freq": ok_freq.tolist(),
            "mean_distance": dsum / dn if dn else 0.0, "n_queries": dn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["magiv2", "magiv3", "transreid"])
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/seed_distance.json"))
    args = ap.parse_args(argv)

    series = ([s.strip() for s in args.series_file.read_text().split("\n") if s.strip()]
              if args.series_file else load_split()["test"])

    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[seeddist] {ckpt} missing, skipping", flush=True); continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in series:
            stream = SeriesStream(args.data_root / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            # gallery-independent: one extraction serves both regimes and every seed
            feats = np.asarray(FT.extract(lm.model, stream, tf, mode="none",
                                          batch_size=BATCH[bb], device=args.device).bn,
                               dtype=np.float64)
            cell: Dict = {}
            for strat in ("random", "temporal"):
                for sd in args.seeds:
                    sm, queries, go = split_seeds(labels, order, args.k, strat, sd)
                    cell.setdefault(strat, {})[str(sd)] = run_cell(feats, labels, order, sm, queries, go)
            per_series[name] = cell
            print(f"[seeddist] {bb} {name} done ({len(order)} crops)", flush=True)
        results[bb] = per_series
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print(f"\n[seeddist] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
