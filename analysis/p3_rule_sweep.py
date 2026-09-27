"""Every P3 decision rule swept over its own parameter grid.

The evaluation runs each rule at one operating point. The paper's appendix reports two sweeps on top of that: the density-aware rule over its minimum core size, and every rule over its grid, both to show that the alternatives relocate the operating point along the purity-versus-fragmentation frontier rather than rising above it. `RULE_GRIDS` in `recognize.protocols.rules` is the single source of the grids, so this sweeps exactly what the framework defines rather than a list retyped here.

P3 is scored on no-memory features, as the protocol specifies.

    python analysis/p3_rule_sweep.py --device cuda --out results/p3_sweep.json
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
from recognize.protocols import run_p3                            # noqa: E402
from recognize.protocols.rules import RULE_GRIDS                  # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
METRICS = ("clusters", "purity", "hungarian", "ari")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["magiv2"])
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/p3_sweep.json"))
    args = ap.parse_args(argv)

    names = load_split()["test"]
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed{args.train_seed}/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[p3sweep] {ckpt} missing, skipping", flush=True)
            continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        acc: Dict[str, Dict[str, List[Dict]]] = {r: {} for r in RULE_GRIDS}
        for name in names:
            stream = SeriesStream(Path("Datasets/popcharacters") / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            f = np.asarray(FT.extract(lm.model, stream, tf, mode="none", batch_size=BATCH[bb],
                                      device=args.device).bn, dtype=np.float64)
            for rule, (param, values) in RULE_GRIDS.items():
                for v in values:
                    acc[rule].setdefault(str(v), []).append(
                        run_p3(f, labels, order, rule=rule, **{param: v}))
            print(f"[p3sweep] {bb} {name} done", flush=True)
        results[bb] = {rule: {v: {m: statistics.fmean(r[m] for r in rs) for m in METRICS}
                              for v, rs in per.items()} for rule, per in acc.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for bb, per in results.items():
        print(f"\n{bb}  (P3, no-memory features, macro over {len(names)} series)")
        print(f"{'rule':<20}{'param':>8}{'#clusters':>11}{'Purity':>9}{'Hung':>8}{'ARI':>8}")
        for rule, vals in per.items():
            param = RULE_GRIDS[rule][0]
            for v, m in vals.items():
                print(f"{rule:<20}{param}={v:<5}{m['clusters']:>11.1f}{100*m['purity']:>9.1f}"
                      f"{100*m['hungarian']:>8.1f}{100*m['ari']:>8.1f}")
    print(f"\n[p3sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
