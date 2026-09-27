"""Evaluation metrics. One retrieval function for P1, P2 and P4.

All values are fractions in [0, 1]. Retrieval is exemplar-level: every gallery
entry of the query's identity is a relevant item. A gallery collapsed to one
entry per identity would leave one relevant item per query, and AP over a single
relevant item is the reciprocal rank, so mAP would equal MRR.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_retrieval_metrics(
    q: np.ndarray, ql: np.ndarray, g: np.ndarray, gl: np.ndarray, ranks: Iterable[int] = (1, 5, 10),
    *, sims: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """mAP, MRR and CMC@k for unit-norm query/gallery features under cosine similarity.

    Queries with no relevant gallery entry are skipped and counted in `n_skipped`.
    Rank positions larger than the gallery are clipped to the gallery size.

    `sims` supplies the (n_query, n_gallery) score matrix instead of `q @ g.T`, for a ranking that is
    not one dot product (a fusion of two feature spaces, say). `q` and `g` are then used only for
    their shapes, so pass the arrays the scores were built from. Everything downstream of the scores
    is the same code, which is the point: a diagnostic must not reimplement the metric.
    """
    q, g = np.asarray(q, dtype=np.float64), np.asarray(g, dtype=np.float64)
    ql, gl = np.asarray(ql), np.asarray(gl)
    ranks = tuple(int(r) for r in ranks)
    n_g = len(gl)
    if sims is None:
        sims = q @ g.T
    else:
        sims = np.asarray(sims, dtype=np.float64)
        if sims.shape != (len(ql), n_g):
            raise ValueError(f"sims has shape {sims.shape}, expected {(len(ql), n_g)}")
    aps, rrs = [], []
    hits_at = {r: 0 for r in ranks}
    skipped = 0
    for i in range(len(ql)):
        rel = (gl == ql[i])
        n_rel = int(rel.sum())
        if n_rel == 0:
            skipped += 1
            continue
        order = np.argsort(-sims[i], kind="stable")
        rel_sorted = rel[order].astype(np.float64)
        first = int(np.argmax(rel_sorted))            # first relevant position (exists: n_rel > 0)
        rrs.append(1.0 / (first + 1))
        precision_at = np.cumsum(rel_sorted) / np.arange(1, n_g + 1)
        aps.append(float((precision_at * rel_sorted).sum() / n_rel))
        for r in ranks:
            if first < min(r, n_g):
                hits_at[r] += 1
    n_q = len(aps)
    out: Dict[str, float] = {
        "mAP": float(np.mean(aps)) if n_q else float("nan"),
        "MRR": float(np.mean(rrs)) if n_q else float("nan"),
        "n_queries": n_q,
        "n_skipped": skipped,
    }
    for r in ranks:
        out[f"R{r}"] = hits_at[r] / n_q if n_q else float("nan")
    return out


def _contingency(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    _, ti = np.unique(true, return_inverse=True)
    _, pi = np.unique(pred, return_inverse=True)
    m = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(m, (ti, pi), 1)
    return m


def clustering_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Predicted cluster count, Purity, NMI, ARI and Hungarian-matched accuracy."""
    true, pred = np.asarray(true), np.asarray(pred)
    m = _contingency(true, pred)
    n = m.sum()
    r, c = linear_sum_assignment(-m)
    return {
        "clusters": int(len(np.unique(pred))),
        "purity": float(m.max(axis=0).sum() / n),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "ari": float(adjusted_rand_score(true, pred)),
        "hungarian": float(m[r, c].sum() / n),
    }
