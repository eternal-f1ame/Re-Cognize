"""The append decision driven by a link constraint instead of a pairwise score.

The three per-pair gates of `append_gate.py` (confidence, margin, reciprocity) fail or barely work, and what they share is that each scores one query-exemplar pair on its own. MagiV2's per-page character-character clustering is relational instead, and on the test set a must-link edge is right 89.9 % of the time (3,977 pairs) against a 40.9 % break-even append precision.

The rule: commit a crop to identity c when it shares a link group with a crop already in the gallery under c, rather than when its own top-1 similarity clears a threshold. Only crops already in the gallery are consulted, so it stays causal. Two variants, abstaining when there is no link (`mustlink`) or falling back to the top-1 (`mustlink+top1`), are compared against frozen, predicted, oracle and the best per-pair gate. `cluster` instead averages the query with the members of its link group already read and appends under the match of that average.

Link groups come from `results/panels/`, written by `detect_panels.py`; no GPU is needed for them, only for the features.

    python analysis/mustlink_append.py --device cuda --out results/mustlink.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
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
from recognize.protocol_constants import B_MAX                    # noqa: E402
from recognize.protocols import run_p4, split_seeds               # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
METRICS = ("R1_identity", "mAP", "wrong_append", "contamination", "n_appended", "n_abstained",
           "n_queries", "n_linked")


def load_groups(panels_dir: Path, series: str) -> List:
    f = panels_dir / f"{series.replace(' ', '_')}.json"
    if not f.exists():
        raise SystemExit(f"no link groups for {series}: run detect_panels.py first ({f})")
    return json.loads(f.read_text())["magi_cluster_of_crop"]


def policies(groups) -> Dict[str, Dict]:
    return {"frozen": {"update_policy": "frozen"},
            "predicted": {"update_policy": "predicted"},
            "margin@0.15": {"update_policy": "margin", "confidence_threshold": 0.15},
            "mustlink": {"update_policy": "mustlink", "link_groups": groups},
            "mustlink+top1": {"update_policy": "mustlink_predicted", "link_groups": groups},
            "cluster": {"update_policy": "cluster", "link_groups": groups},
            "oracle": {"update_policy": "oracle"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["magiv2", "magiv3"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--series", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--strategy", default="random", choices=("random", "temporal"))
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--panels", type=Path, default=Path("results/panels"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/mustlink.json"))
    args = ap.parse_args(argv)

    names = args.series or load_split()["test"]
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[mustlink] {ckpt} missing, skipping", flush=True)
            continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in names:
            stream = SeriesStream(Path("Datasets/popcharacters") / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            groups = load_groups(args.panels, name)
            if len(groups) != len(labels):
                raise SystemExit(f"{name}: {len(groups)} link groups for {len(labels)} crops")
            tokens = FT.backbone_tokens(lm.model, stream, tf, batch_size=BATCH[bb], device=args.device)
            cell = {}
            for seed in args.seeds:
                seed_map, _, gallery_only = split_seeds(labels, order, args.k, args.strategy, seed)
                g_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
                f = FT.extract(lm.model, stream, tf, mode="memory", gallery_idx=g_idx,
                               batch_size=BATCH[bb], device=args.device, tokens=tokens).bn
                for label, kw in policies(groups).items():
                    r = run_p4(f, labels, order, args.k, args.strategy, seed, b_max=B_MAX, **kw)
                    cell.setdefault(label, {})[str(seed)] = {m: r.get(m) for m in METRICS}
            per_series[name] = cell
            print(f"[mustlink] {bb} {name} done", flush=True)
        results[bb] = per_series

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    for bb, ps in results.items():
        def macro(policy, metric):
            vals = [c[policy][s][metric] for c in ps.values() for s in c[policy]
                    if c[policy][s][metric] is not None]
            return statistics.fmean(vals) if vals else float("nan")
        fz = macro("frozen", "R1_identity") * 100
        print(f"\n{bb}  P4 Seq-{'R' if args.strategy == 'random' else 'T'} at k={args.k}, "
              f"B_max={B_MAX}. Static gallery = {fz:.2f}")
        print(f"{'policy':<16}{'id Rank-1':>11}{'vs static':>11}{'append':>9}{'wrong':>9}"
              f"{'contam':>9}{'by link':>9}")
        for policy in ("frozen", "predicted", "margin@0.15", "mustlink", "mustlink+top1",
                       "cluster", "oracle"):
            v = macro(policy, "R1_identity") * 100
            nq = macro(policy, "n_queries")
            print(f"{policy:<16}{v:>11.2f}{v - fz:>+11.2f}"
                  f"{100 * macro(policy, 'n_appended') / max(1e-9, nq):>8.1f}%"
                  f"{100 * macro(policy, 'wrong_append'):>8.1f}%"
                  f"{100 * macro(policy, 'contamination'):>8.1f}%"
                  f"{100 * macro(policy, 'n_linked') / max(1e-9, nq):>8.1f}%")
    print(f"\n[mustlink] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
