"""Neighbour Feature Centralization on our checkpoints: is reciprocity the filter the memory lacks?

Pose2ID (CVPR 2025) reports +4.04 mAP and +0.35 Rank-1 on Market1501 from a training-free operation: for each feature take its top-k1 neighbours, keep the ones that also rank it inside their own top-k2, sum, renormalise. Our learned memory does the same kind of aggregation over a FIFO routed by the model's top-1 guess, and gets +0.5 mAP and -1.0 Rank-1. The difference is what gets aggregated, so this measures the two on the same features.

    centralized_i = normalize( z_i + sum_{j in M_i} z_j ),   M_i = { j in kNN_k1(i) : i in kNN_k2(j) }

Four rows per backbone: the finetuned checkpoint's features plain and centralized, and the memory checkpoint's features plain and centralized. The last one asks whether the two compose or overlap.

The neighbour sets are also scored directly: what fraction of each M_i shares the anchor's identity. That is the quantity to compare against the memory's routing accuracy, which is 9-24 % (TransReID) and 27-61 % (MagiV2).

Transductive: neighbours come from the whole series, which is standard for re-ranking in Re-ID and is available to P1 and P2 since the stream is given. It would NOT be legitimate for P4, whose decisions are causal; that variant needs neighbours restricted to already-seen crops.

    python analysis/nfc.py --device cuda --out results/nfc.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402
from recognize.protocols import p1_split, run_p1, run_p2, split_seeds  # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
GRID = [(2, 2), (4, 2), (4, 4), (8, 4), (8, 8), (16, 8)]
SIDES = ("both", "query", "gallery")


def mutual_neighbours(feats: np.ndarray, k1: int, k2: int) -> List[np.ndarray]:
    """M_i = the top-k1 neighbours of i that also keep i inside their own top-k2."""
    sims = feats @ feats.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1, kind="stable")
    top1, top2 = order[:, :k1], order[:, :k2]
    keeps_me = np.zeros(sims.shape, dtype=bool)
    rows = np.repeat(np.arange(len(feats)), top2.shape[1])
    keeps_me[rows, top2.ravel()] = True                 # keeps_me[j, i] = i is in j's top-k2
    return [cand[keeps_me[cand, i]] for i, cand in enumerate(top1)]


def centralize(feats: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """Pose2ID's NFC: add the mutual neighbours, renormalise. A no-op where M_i is empty."""
    out = feats.copy()
    for i, nb in enumerate(mutual_neighbours(feats, k1, k2)):
        if len(nb):
            out[i] = feats[i] + feats[nb].sum(axis=0)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def neighbour_purity(feats: np.ndarray, labels: np.ndarray, k1: int, k2: int) -> Dict[str, float]:
    """How often an aggregated neighbour actually shares the anchor's identity."""
    M = mutual_neighbours(feats, k1, k2)
    hits = tot = empty = 0
    sizes = []
    for i, nb in enumerate(M):
        sizes.append(len(nb))
        if not len(nb):
            empty += 1
            continue
        hits += int((labels[nb] == labels[i]).sum())
        tot += len(nb)
    return {"purity": hits / tot if tot else float("nan"),
            "mean_size": float(np.mean(sizes)), "empty_frac": empty / len(M)}


def blend(plain: np.ndarray, central: np.ndarray, idx, side: str) -> np.ndarray:
    """Replace only the chosen side's rows with their centralized version.

    Pose2ID reports query-only, gallery-only and both separately, and the distinction matters far more here than on Market1501: at P2 with k=1 a gallery is one crop per identity, so centralizing it averages away the only exemplar that identity has.
    """
    if side == "both":
        return central
    out = plain.copy()
    out[np.asarray(idx, dtype=int)] = central[np.asarray(idx, dtype=int)]
    return out


def evaluate(plain: np.ndarray, central: np.ndarray, labels, order,
             seeds: Sequence[int], side: str) -> Dict[str, float]:
    p1, p2 = [], []
    for s in seeds:
        g1, q1, _ = p1_split(labels, s)
        f1 = blend(plain, central, q1 if side == "query" else g1, side)
        p1.append(run_p1(f1, labels, s))
        seed_map, q2, gallery_only = split_seeds(labels, order, 1, "random", s)
        g2 = sorted([i for v in seed_map.values() for i in v]
                    + [i for v in gallery_only.values() for i in v])
        f2 = blend(plain, central, q2 if side == "query" else g2, side)
        p2.append(run_p2(f2, labels, order, 1, "random", s))
    return {"p1_mAP": 100 * statistics.fmean(m["mAP"] for m in p1),
            "p1_R1": 100 * statistics.fmean(m["R1"] for m in p1),
            "p2_mAP": 100 * statistics.fmean(m["mAP"] for m in p2),
            "p2_R1": 100 * statistics.fmean(m["R1"] for m in p2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["magiv2", "magiv3", "transreid"])
    ap.add_argument("--configs", nargs="+", default=["finetuned", "memory"])
    ap.add_argument("--series", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/nfc.json"))
    args = ap.parse_args(argv)

    series_names = args.series or load_split()["test"]
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        for cfg in args.configs:
            ckpt = Path(f"checkpoints/{bb}/{cfg}/seed0/{args.checkpoint_name}")
            if not ckpt.exists():
                print(f"[nfc] {ckpt} missing, skipping", flush=True)
                continue
            lm = load_checkpoint(str(ckpt), device=args.device)
            per_series: Dict = {}
            for name in series_names:
                stream = SeriesStream(Path("Datasets/popcharacters") / name)
                labels, order = np.asarray(stream.labels), stream.reading_order
                tokens = FT.backbone_tokens(lm.model, stream, tf, batch_size=BATCH[bb], device=args.device)
                plain = FT.extract(lm.model, stream, tf, mode="none", batch_size=BATCH[bb],
                                   device=args.device, tokens=tokens).bn
                cell = {"plain": evaluate(plain, plain, labels, order, args.seeds, "both")}
                for k1, k2 in GRID:
                    central = centralize(plain, k1, k2)
                    for side in SIDES:
                        cell[f"nfc_{side}_{k1}_{k2}"] = evaluate(plain, central, labels, order,
                                                                 args.seeds, side)
                    cell[f"purity_{k1}_{k2}"] = neighbour_purity(plain, labels, k1, k2)
                per_series[name] = cell
                print(f"[nfc] {bb} {cfg} {name} done", flush=True)
            results[f"{bb}_{cfg}"] = per_series

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    for key, per_series in results.items():
        def macro(row, metric):
            vals = [s[row][metric] for s in per_series.values() if row in s]
            return statistics.fmean(vals) if vals else float("nan")
        base = {m: macro("plain", m) for m in ("p1_mAP", "p1_R1", "p2_mAP", "p2_R1")}
        print(f"\n{key}: plain P1 {base['p1_mAP']:.2f}/{base['p1_R1']:.2f}  "
              f"P2-R@1 {base['p2_mAP']:.2f}/{base['p2_R1']:.2f}")
        print(f"{'side':>8}{'k1,k2':>8}{'dP1 mAP':>9}{'dP1 R1':>9}{'dP2 mAP':>9}{'dP2 R1':>9}"
              f"{'|M|':>7}{'purity':>9}{'empty':>8}")
        for side in SIDES:
            for k1, k2 in GRID:
                row, pur = f"nfc_{side}_{k1}_{k2}", f"purity_{k1}_{k2}"
                print(f"{side:>8}{f'{k1},{k2}':>8}"
                      + "".join(f"{macro(row, m) - base[m]:>+9.2f}"
                                for m in ("p1_mAP", "p1_R1", "p2_mAP", "p2_R1"))
                      + f"{macro(pur, 'mean_size'):>7.1f}{100 * macro(pur, 'purity'):>8.1f}%"
                      + f"{100 * macro(pur, 'empty_frac'):>7.1f}%")
    print(f"\n[nfc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
