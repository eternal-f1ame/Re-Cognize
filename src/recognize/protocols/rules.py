"""P3 online decision rules.

Each rule consumes L2-normalised features in reading order and returns one
cluster id per crop. Centroids are running means re-normalised after every
update. `fixed` at tau=0.55 is the paper's reference rule; the other four are
alternative instantiations for the P3 diagnostic, not proposed methods.
"""
from __future__ import annotations

import numpy as np


def fixed(F, tau=0.55):
    """Paper's reference rule: join the nearest centroid above a global tau."""
    cents, counts, out = [], [], []
    for f in F:
        if cents:
            sims = np.array([c @ f for c in cents])
            j = int(sims.argmax())
            if sims[j] >= tau:
                cents[j] = (cents[j] * counts[j] + f)
                cents[j] /= np.linalg.norm(cents[j]) + 1e-8
                counts[j] += 1
                out.append(j)
                continue
        cents.append(f.copy()); counts.append(1); out.append(len(cents) - 1)
    return np.array(out)


def variance_adaptive(F, lam=2.0, tau0=0.55):
    """Per-cluster tau_c = mu_c - lam * sigma_c over observed join similarities."""
    cents, counts, sims_log, out = [], [], [], []
    for f in F:
        joined = False
        if cents:
            sims = np.array([c @ f for c in cents])
            j = int(sims.argmax())
            s = sims_log[j]
            thr = (np.mean(s) - lam * np.std(s)) if len(s) >= 2 else tau0
            if sims[j] >= thr:
                cents[j] = (cents[j] * counts[j] + f)
                cents[j] /= np.linalg.norm(cents[j]) + 1e-8
                counts[j] += 1; sims_log[j].append(float(sims[j]))
                out.append(j); joined = True
        if not joined:
            cents.append(f.copy()); counts.append(1); sims_log.append([])
            out.append(len(cents) - 1)
    return np.array(out)


def density_aware(F, n_min=5, tau0=0.55):
    """DBSCAN-style: clusters with >= n_min members use their median core
    similarity as the threshold; smaller ones fall back to tau0."""
    cents, counts, sims_log, out = [], [], [], []
    for f in F:
        joined = False
        if cents:
            sims = np.array([c @ f for c in cents])
            j = int(sims.argmax())
            s = sims_log[j]
            thr = float(np.median(s)) if counts[j] >= n_min and s else tau0
            if sims[j] >= thr:
                cents[j] = (cents[j] * counts[j] + f)
                cents[j] /= np.linalg.norm(cents[j]) + 1e-8
                counts[j] += 1; sims_log[j].append(float(sims[j]))
                out.append(j); joined = True
        if not joined:
            cents.append(f.copy()); counts.append(1); sims_log.append([])
            out.append(len(cents) - 1)
    return np.array(out)


def cohesion_relative(F, alpha=0.9, tau0=0.55):
    """Threshold proportional to the cluster's running mean similarity."""
    cents, counts, sims_log, out = [], [], [], []
    for f in F:
        joined = False
        if cents:
            sims = np.array([c @ f for c in cents])
            j = int(sims.argmax())
            s = sims_log[j]
            thr = alpha * float(np.mean(s)) if s else tau0
            if sims[j] >= thr:
                cents[j] = (cents[j] * counts[j] + f)
                cents[j] /= np.linalg.norm(cents[j]) + 1e-8
                counts[j] += 1; sims_log[j].append(float(sims[j]))
                out.append(j); joined = True
        if not joined:
            cents.append(f.copy()); counts.append(1); sims_log.append([])
            out.append(len(cents) - 1)
    return np.array(out)


def graph_louvain(F, knn=5, reading_order_edges=True, seed=42):
    """Louvain over a mutual-kNN graph, with edges between adjacent crops in
    reading order. Mutual means an edge needs reciprocity, so a rare crop
    cannot unilaterally bridge to another identity."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    n = len(F)
    S = F @ F.T
    np.fill_diagonal(S, -np.inf)
    k = min(knn, max(1, n - 1))
    nbr = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    nbrset = [set(row.tolist()) for row in nbr]

    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n):
        for j in nbrset[i]:
            if i in nbrset[j]:                       # mutual kNN only
                G.add_edge(i, int(j), weight=float(max(S[i, j], 0.0)))
    if reading_order_edges:
        for i in range(n - 1):
            w = float(max(S[i, i + 1], 0.0))
            if G.has_edge(i, i + 1):
                G[i][i + 1]["weight"] = max(G[i][i + 1]["weight"], w)
            else:
                G.add_edge(i, i + 1, weight=w)

    comms = louvain_communities(G, weight="weight", seed=seed)
    out = np.zeros(n, dtype=int)
    for cid, nodes in enumerate(comms):
        for v in nodes:
            out[v] = cid
    return out


RULES = {
    "fixed": fixed,
    "variance-adaptive": variance_adaptive,
    "density-aware": density_aware,
    "cohesion-relative": cohesion_relative,
    "graph-louvain": graph_louvain,
}

# Sweep grids for the rule comparison in analysis/p3_rule_sweep.py (kept with the rules).
RULE_GRIDS = {
    "fixed": ("tau", [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]),
    "variance-adaptive": ("lam", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    "density-aware": ("n_min", [2, 3, 5, 8, 12, 20]),
    "cohesion-relative": ("alpha", [0.70, 0.80, 0.85, 0.90, 0.95, 0.99]),
    "graph-louvain": ("knn", [2, 3, 5, 8, 10, 15]),
}
