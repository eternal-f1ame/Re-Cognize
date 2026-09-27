"""P1 closed-set retrieval."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..metrics import compute_retrieval_metrics
from ..protocol_constants import GALLERY_RATIO


def p1_split(labels: Sequence[int], seed: int, gallery_ratio: float = GALLERY_RATIO) -> Tuple[List[int], List[int], int]:
    """(gallery indices, query indices, n_excluded_singletons) for P1 at `seed`."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    g_idx, q_idx, singletons = [], [], 0
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) < 2:
            singletons += 1
            continue
        n_g = max(1, int(np.floor(gallery_ratio * len(idx))))
        chosen = set(int(i) for i in rng.choice(idx, size=n_g, replace=False))
        g_idx.extend(sorted(chosen))
        q_idx.extend(int(i) for i in idx if int(i) not in chosen)
    return sorted(g_idx), sorted(q_idx), singletons


def run_p1(features: np.ndarray, labels: Sequence[int], seed: int, gallery_ratio: float = GALLERY_RATIO) -> Dict:
    """Per identity: gallery = max(1, floor(gallery_ratio * n)) random crops, the rest queries.

    Identities with a single crop are excluded and counted in `n_excluded_singletons`.
    """
    labels = np.asarray(labels)
    g_idx, q_idx, singletons = p1_split(labels, seed, gallery_ratio)
    out = compute_retrieval_metrics(features[q_idx], labels[q_idx], features[g_idx], labels[g_idx])
    out.update({
        "n_identities": int(len(np.unique(labels)) - singletons),
        "n_excluded_singletons": int(singletons),
        "n_gallery": len(g_idx),
        "gallery_ratio": float(gallery_ratio),
        "seed": int(seed),
    })
    return out
