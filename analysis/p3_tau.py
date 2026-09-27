"""P3 at more than one novelty threshold, so the fragmentation row keeps its evidence.

The evaluation runs P3 only at the protocol's reference threshold, tau_nov = 0.55. The paper also reports a loose threshold to make a measurement point that a single operating point cannot: Purity rises monotonically as clusters fragment, so Purity alone is gameable and has to be read against the cluster count. This computes P3 at each threshold given (by default 0.55 and 0.80).

P3 is scored on no-memory features, as the protocol specifies, so the row is a property of the backbone rather than of the memory block.

    python analysis/p3_tau.py --device cuda --out results/p3_tau.json
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

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
METRICS = ("clusters", "purity", "hungarian", "ari", "nmi")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["transreid", "magiv2"])
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.55, 0.80])
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/p3_tau.json"))
    args = ap.parse_args(argv)

    names = load_split()["test"]
    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed{args.train_seed}/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[p3tau] {ckpt} missing, skipping", flush=True)
            continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_tau: Dict[str, List[Dict]] = {f"{t:.2f}": [] for t in args.taus}
        for name in names:
            stream = SeriesStream(Path("Datasets/popcharacters") / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            f = np.asarray(FT.extract(lm.model, stream, tf, mode="none",
                                      batch_size=BATCH[bb], device=args.device).bn, dtype=np.float64)
            for t in args.taus:
                per_tau[f"{t:.2f}"].append(run_p3(f, labels, order, rule="fixed", tau=t))
            print(f"[p3tau] {bb} {name} done", flush=True)
        results[bb] = {t: {m: statistics.fmean(r[m] for r in rs) for m in METRICS}
                       for t, rs in per_tau.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n{'backbone':<13}{'tau':>6}{'#clusters':>11}{'Purity':>9}{'Hung':>8}{'ARI':>8}")
    for bb, per in results.items():
        for t, m in per.items():
            print(f"{bb:<13}{t:>6}{m['clusters']:>11.1f}{100*m['purity']:>9.1f}"
                  f"{100*m['hungarian']:>8.1f}{100*m['ari']:>8.1f}")
    print(f"\n[p3tau] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
