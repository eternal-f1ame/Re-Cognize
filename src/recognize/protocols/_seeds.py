"""Seed / query split shared by P2 and P4."""
from __future__ import annotations

from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np


def split_seeds(
    labels: Sequence[int],
    order: Sequence[int],
    k: int,
    strategy: Literal["random", "temporal"],
    seed: int,
) -> Tuple[Dict[int, List[int]], List[int], Dict[int, List[int]]]:
    """Return (seed_map, query_indices, gallery_only).

    seed_map: identity -> k seed crop indices (Seq-T: first k in reading order;
    Seq-R: k drawn without replacement by `np.random.default_rng(seed)`, identities
    visited in sorted label order). query_indices: every non-seed crop of a seeded
    identity, in reading order. gallery_only: identities with <= k crops -> all their
    indices (they enter the gallery as distractors and never appear as queries).
    """
    if strategy not in ("random", "temporal"):
        raise ValueError(f"strategy must be 'random' or 'temporal', got {strategy!r}")
    labels = np.asarray(labels)
    order = [int(i) for i in order]
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[int]] = {}
    for i in order:
        by_label.setdefault(int(labels[i]), []).append(i)
    seed_map: Dict[int, List[int]] = {}
    gallery_only: Dict[int, List[int]] = {}
    seed_set = set()
    for lab in sorted(by_label):
        idx = by_label[lab]
        if len(idx) <= k:
            gallery_only[lab] = list(idx)
            seed_set.update(idx)
            continue
        if strategy == "temporal":
            seeds = list(idx[:k])
        else:
            seeds = sorted(int(j) for j in rng.choice(np.asarray(idx), size=k, replace=False))
            seeds = [j for j in idx if j in set(seeds)]          # keep reading order
        seed_map[lab] = seeds
        seed_set.update(seeds)
    query_indices = [i for i in order if i not in seed_set]
    return seed_map, query_indices, gallery_only
