"""Dev-set model selection through the evaluation harness.

`best.pth` is chosen on the dev series only: P1 mAP (seed 0) for models without
memory, P2 Seq-R@k=1 mAP (seed 0) for models with memory, both computed exactly as
the final evaluation computes them (SeriesStream, extract, run_p1 / run_p2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .data import SeriesStream
from .features import extract
from .protocols import run_p1, run_p2, split_seeds


def dev_score(model, transform, series_dirs: Sequence[Path], *, memory: bool, device: str = "cpu",
              batch_size: int = 64, num_workers: int = 0, both_metrics: bool = True) -> Dict:
    """Macro dev metric over `series_dirs`; leaves the model's memory block and train/eval mode untouched.

    Selection uses P1 mAP without memory and P2 Seq-R@1 mAP with it, so the selection metric of a
    memory run is NOT comparable with that of a no-memory run. `both_metrics` additionally records
    the no-memory P1 and P2@1 of the same weights under `also`, which are comparable across configs.
    """
    was_training = getattr(model, "training", False)
    original_block = getattr(model, "memory_block", None)
    per: Dict[str, float] = {}
    also: Dict[str, Dict[str, float]] = {}
    try:
        for d in series_dirs:
            s = SeriesStream(Path(d))
            kw = dict(batch_size=batch_size, device=device, num_workers=num_workers)
            if memory:
                seed_map, _, gallery_only = split_seeds(s.labels, s.reading_order, 1, "random", 0)
                g_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
                f = extract(model, s, transform, mode="memory", gallery_idx=g_idx, **kw)
                r = run_p2(f.bn, s.labels, s.reading_order, 1, "random", 0)
                metric = "p2r1_map"
            else:
                f = extract(model, s, transform, mode="none", **kw)
                r = run_p1(f.bn, s.labels, 0)
                metric = "p1_map"
            per[s.name] = float(r["mAP"])
            if both_metrics:
                plain = extract(model, s, transform, mode="none", **kw)
                also.setdefault("nomem_p1_map", {})[s.name] = float(run_p1(plain.bn, s.labels, 0)["mAP"])
                also.setdefault("nomem_p2r1_map", {})[s.name] = float(
                    run_p2(plain.bn, s.labels, s.reading_order, 1, "random", 0)["mAP"])
    finally:
        # extract(mode="memory") re-initialises the memory block for the dev identities; the optimizer
        # still holds the training block's parameters, so put that block back.
        if original_block is not None and getattr(model, "memory_block", None) is not original_block:
            model.memory_block = original_block
        if was_training:
            model.train()
    out = {"metric": metric, "value": float(np.mean(list(per.values()))) if per else float("nan"), "per_series": per}
    for key, values in also.items():
        out[key] = float(np.mean(list(values.values()))) if values else float("nan")
    return out
