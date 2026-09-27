"""Which crops to append to a growing P4 gallery: can a gate on the top-1 decision beat appending every one?

On Seq-R at k=1, appends under the true label (oracle) lift identity Rank-1 by 22 to 27 points over the static gallery, while appends by the model's own top-1 lose between 0.4 and 3.6. That gap is far larger than what the representation moves: the memory block itself moves P1 mAP by about half a point. So the question is not how to represent a crop better, it is when to trust a label enough to keep it.

This sweeps the cheapest answers across the whole test set, against the three reference policies: append only when the top-1 cosine clears a threshold (`confident`), only when the top-1 beats the best other identity by a margin (`margin`), or only when the query is also among the matched exemplar's k nearest crops seen so far (`reciprocal`). `frozen` is the floor a gate must beat (it is the static gallery) and `oracle` the ceiling.

    python analysis/append_gate.py --device cuda --out results/append_gate.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

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
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MARGINS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
RECIPROCAL_K = [1, 2, 4, 8, 16, 32]
METRICS = ("R1_identity", "mAP", "wrong_append", "contamination", "n_appended", "n_abstained", "n_queries")


def run_series(model, stream, transform, batch: int, seeds, device: str, k: int, strategy: str) -> dict:
    """Every policy and threshold on one series, from one backbone pass."""
    tokens = FT.backbone_tokens(model, stream, transform, batch_size=batch, device=device)
    labels, order = np.asarray(stream.labels), stream.reading_order
    out = {}
    for seed in seeds:
        seed_map, _, gallery_only = split_seeds(labels, order, k, strategy, seed)
        g_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
        f = FT.extract(model, stream, transform, mode="memory", gallery_idx=g_idx,
                       batch_size=batch, device=device, tokens=tokens).bn
        cell = {}
        for policy in ("predicted", "oracle", "frozen"):
            cell[policy] = run_p4(f, labels, order, k, strategy, seed, b_max=B_MAX, update_policy=policy)
        for tau in THRESHOLDS:
            cell[f"confident@{tau:.1f}"] = run_p4(f, labels, order, k, strategy, seed, b_max=B_MAX,
                                                  update_policy="confident", confidence_threshold=tau)
        for m in MARGINS:
            cell[f"margin@{m:.2f}"] = run_p4(f, labels, order, k, strategy, seed, b_max=B_MAX,
                                             update_policy="margin", confidence_threshold=m)
        for kk in RECIPROCAL_K:
            cell[f"recip@{kk}"] = run_p4(f, labels, order, k, strategy, seed, b_max=B_MAX,
                                         update_policy="reciprocal", reciprocal_k=kk)
        out[str(seed)] = {name: {m: r[m] for m in METRICS} for name, r in cell.items()}
    return out


def macro(per_series: dict, policy: str, metric: str) -> float:
    by_seed = {}
    for series in per_series.values():
        for seed, cell in series.items():
            if policy in cell:
                by_seed.setdefault(seed, []).append(cell[policy][metric])
    return float(np.mean([np.mean(v) for v in by_seed.values()])) if by_seed else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["transreid", "magiv2", "magiv3"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--series", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--strategy", default="random", choices=("random", "temporal"))
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/append_gate.json"))
    args = ap.parse_args(argv)

    series_names = args.series or load_split()["test"]
    results = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[append_gate] {ckpt} does not exist, skipping", flush=True)
            continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series = {}
        for name in series_names:
            stream = SeriesStream(Path("Datasets/popcharacters") / name)
            per_series[name] = run_series(lm.model, stream, tf, BATCH[bb], args.seeds,
                                          args.device, args.k, args.strategy)
            print(f"[append_gate] {bb} {name} done", flush=True)
        results[bb] = {"checkpoint": str(ckpt), "per_series": per_series}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    for bb, r in results.items():
        ps = r["per_series"]
        frozen = macro(ps, "frozen", "R1_identity") * 100
        print(f"\n{bb}  P4 Seq-{'R' if args.strategy == 'random' else 'T'} at k={args.k}, B_max={B_MAX}, "
              f"identity Rank-1. Static gallery (frozen) = {frozen:.2f}")
        print(f"{'policy':<16}{'id Rank-1':>11}{'vs frozen':>11}{'append rate':>13}{'wrong appends':>15}"
              f"{'contamination':>15}")
        rows = (["frozen", "predicted"] + [f"confident@{t:.1f}" for t in THRESHOLDS]
                + [f"margin@{m:.2f}" for m in MARGINS]
                + [f"recip@{kk}" for kk in RECIPROCAL_K] + ["oracle"])
        for policy in rows:
            v = macro(ps, policy, "R1_identity") * 100
            napp, nq = macro(ps, policy, "n_appended"), macro(ps, policy, "n_queries")
            print(f"{policy:<16}{v:>11.2f}{v - frozen:>+11.2f}{100 * napp / nq if nq else 0:>12.1f}%"
                  f"{100 * macro(ps, policy, 'wrong_append'):>14.1f}%"
                  f"{100 * macro(ps, policy, 'contamination'):>14.1f}%")
        gates = ([f"confident@{t:.1f}" for t in THRESHOLDS] + [f"margin@{m:.2f}" for m in MARGINS]
                 + [f"recip@{kk}" for kk in RECIPROCAL_K])
        best = max(gates, key=lambda p: macro(ps, p, "R1_identity"))
        print(f"  best gate: {best} at {macro(ps, best, 'R1_identity') * 100:.2f}, "
              f"{macro(ps, best, 'R1_identity') * 100 - frozen:+.2f} against the static gallery and "
              f"{macro(ps, 'oracle', 'R1_identity') * 100 - frozen:+.2f} available with true labels")
    print(f"\n[append_gate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
