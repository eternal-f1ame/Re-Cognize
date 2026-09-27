"""P3 online clustering over the full stream in reading order."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..metrics import clustering_metrics
from .rules import RULES


def run_p3(features: np.ndarray, labels: Sequence[int], order: Sequence[int], rule: str = "fixed", **params) -> Dict:
    """Assign every crop online with `rule`; cluster count is unbounded."""
    if rule not in RULES:
        raise KeyError(f"unknown P3 rule {rule!r}; choose from {tuple(RULES)}")
    labels = np.asarray(labels)
    order = np.asarray([int(i) for i in order])
    F = np.asarray(features, dtype=np.float64)[order]
    pred_in_order = RULES[rule](F, **params)
    pred = np.empty(len(labels), dtype=int)
    pred[order] = pred_in_order
    out = clustering_metrics(labels, pred)
    out.update({"rule": rule, "params": dict(params), "n_crops": int(len(labels)),
                "n_identities": int(len(np.unique(labels)))})
    return out
