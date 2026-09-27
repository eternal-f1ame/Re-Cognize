"""P2 seeded static gallery."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..metrics import compute_retrieval_metrics
from ._seeds import split_seeds


def run_p2(features: np.ndarray, labels: Sequence[int], order: Sequence[int], k: int, strategy: str, seed: int) -> Dict:
    """Gallery = k seeds per identity plus every crop of gallery-only identities; queries = the rest."""
    labels = np.asarray(labels)
    seed_map, queries, gallery_only = split_seeds(labels, order, k, strategy, seed)
    g_idx = sorted([i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v])
    out = compute_retrieval_metrics(features[queries], labels[queries], features[g_idx], labels[g_idx])
    out.update({
        "n_gallery_only_identities": len(gallery_only),
        "n_seeded_identities": len(seed_map),
        "n_gallery": len(g_idx),
        "k": int(k), "strategy": strategy, "seed": int(seed),
    })
    return out
