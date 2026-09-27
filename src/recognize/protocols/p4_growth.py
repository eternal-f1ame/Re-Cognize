"""P4 online gallery growth.

Starts as P2. Queries are processed in reading order; each is scored against the
gallery snapshot taken before its own update (exemplar-level AP, same metric
function as P1/P2), then appended to one identity's FIFO buffer according to
`update_policy`: "predicted" (top-1 identity), "oracle" (true identity),
"frozen" (no growth), or "confident" (the top-1 identity, but only when the top-1
cosine reaches `confidence_threshold`). Seeds are protected; `b_max` bounds the
growth buffer per identity (None = unbounded).

"confident" exists because the gap between oracle and predicted appends is the one
large effect in P4: on Seq-R at k=1, oracle appends lift identity Rank-1 by 22 to 27
points over the static gallery, while predicted appends lose 0.4 to 3.6. All the
headroom in P4 is in deciding what to append, and abstaining is the cheapest
decision available.

"assign" is the only policy that does not decide one crop at a time: it holds a page's
queries until the page ends, then assigns the page's must-link groups to gallery identities
jointly, under mutual exclusion and a reject option (`assign.py`). Deferring costs it the
within-page appends the other policies get, and buys the page-level structure that the
per-crop rules have no way to use.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..metrics import compute_retrieval_metrics
from ..protocol_constants import B_MAX
from ._seeds import split_seeds
from .assign import ETA_DEFAULT, assign_page, page_groups

UPDATE_POLICIES = ("predicted", "oracle", "frozen", "confident", "margin", "reciprocal",
                   "mustlink", "mustlink_predicted", "cluster", "assign")
GATED_POLICIES = ("confident", "margin")
LINK_POLICIES = ("mustlink", "mustlink_predicted", "cluster")
GROUP_POLICIES = LINK_POLICIES + ("assign",)


def _linked_identity(link_groups: Sequence, g_src: np.ndarray, g_ident: np.ndarray,
                     active: np.ndarray, n_used: int, qi: int):
    """The identity of the already-assigned crops that share this query's link group, or None.

    A link group is MagiV2's per-page character-character clustering, measured at 89.9 % precision
    over 3,977 within-page pairs on the test set: more than twice the 40.9 % break-even append
    precision, and relational rather than pairwise, which is what the three per-pair gates lack.
    Only crops already in the gallery are consulted, so the rule stays causal.
    """
    grp = link_groups[qi]
    if grp is None:
        return None
    votes: Dict[int, int] = {}
    for r in range(n_used):
        if not active[r]:
            continue
        src = int(g_src[r])
        if src < 0 or link_groups[src] != grp:
            continue
        votes[int(g_ident[r])] = votes.get(int(g_ident[r]), 0) + 1
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def _is_reciprocal(features: np.ndarray, seen_idx: Sequence[int], qi: int,
                   exemplar: np.ndarray, k: int) -> bool:
    """Is the query among the exemplar's k nearest neighbours, over the crops seen so far?

    The top-1 test asks "is this exemplar the query's nearest neighbour". This asks the other half:
    of everything the stream has delivered, is the query one of the k closest to that exemplar. Both
    directions have to hold for the pair to be mutual. Reciprocity is a far better filter than the
    model's own top-1: on MagiV2, 58-61 % of mutual neighbours share the anchor's identity, against
    27-61 % for the FIFO's routing, and the break-even append precision is 40.9 %.
    """
    cand = np.asarray(list(seen_idx) + [qi], dtype=int)
    sims = features[cand] @ exemplar
    return int((sims > sims[-1]).sum()) < k


def _gate_score(policy: str, sims: np.ndarray, gallery_ident: np.ndarray, pred: int) -> float:
    """How much the top-1 decision is worth trusting.

    "confident": the top-1 cosine itself. "margin": the top-1 cosine minus the best cosine to any
    *other* identity, which is the quantity that actually separates a decision from its runner-up.
    A raw similarity says how close the nearest neighbour is, not how much closer it is than the
    alternative. If no other identity is present the margin is +inf: there is nothing to confuse it
    with.
    """
    top = float(sims.max())
    if policy == "confident":
        return top
    other = sims[gallery_ident != pred]
    return float("inf") if other.size == 0 else top - float(other.max())


def run_p4(
    features: np.ndarray,
    labels: Sequence[int],
    order: Sequence[int],
    k: int,
    strategy: str,
    seed: int,
    b_max: Optional[int] = B_MAX,
    update_policy: str = "predicted",
    confidence_threshold: Optional[float] = None,
    reciprocal_k: Optional[int] = None,
    link_groups: Optional[Sequence] = None,
    page_ids: Optional[Sequence] = None,
    eta: float = ETA_DEFAULT,
    unlinked: str = "abstain",
    exclusive: bool = True,
) -> Dict:
    if update_policy not in UPDATE_POLICIES:
        raise ValueError(f"update_policy must be one of {UPDATE_POLICIES}, got {update_policy!r}")
    if (update_policy in GATED_POLICIES) != (confidence_threshold is not None):
        raise ValueError(f"confidence_threshold belongs to update_policy in {GATED_POLICIES} and only to those")
    if (update_policy == "reciprocal") != (reciprocal_k is not None):
        raise ValueError("reciprocal_k belongs to update_policy='reciprocal' and only to it")
    if (update_policy in GROUP_POLICIES) != (link_groups is not None):
        raise ValueError(f"link_groups belongs to update_policy in {GROUP_POLICIES} and only to those")
    if (update_policy == "assign") != (page_ids is not None):
        raise ValueError("page_ids belongs to update_policy='assign' and only to it")
    labels = np.asarray(labels)
    features = np.asarray(features, dtype=np.float64)
    seed_map, queries, gallery_only = split_seeds(labels, order, k, strategy, seed)
    seed_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]

    # Gallery storage: rows appended once, evictions flip `active`.
    cap = len(seed_idx) + len(queries)
    g_feats = np.zeros((cap, features.shape[1]))
    g_ident = np.zeros(cap, dtype=labels.dtype)       # identity the exemplar is filed under
    g_true = np.zeros(cap, dtype=labels.dtype)        # true label of the exemplar (for contamination)
    g_src = np.full(cap, -1, dtype=int)               # crop each exemplar came from, for link lookup
    active = np.zeros(cap, dtype=bool)
    n_used = 0
    buffers: Dict[int, deque] = {}
    for i in sorted(seed_idx):
        g_feats[n_used], g_ident[n_used], g_true[n_used], active[n_used] = features[i], labels[i], labels[i], True
        g_src[n_used] = i
        n_used += 1
    for lab in list(seed_map) + list(gallery_only):
        buffers[int(lab)] = deque()

    per_query, preds, trues = [], [], []
    wrong = appended = abstained = linked_used = 0

    def commit(qi: int, target: int) -> None:
        """File one crop under `target`, evicting the identity's oldest growth exemplar if full."""
        nonlocal n_used, appended, wrong
        if target not in buffers:
            return
        buf = buffers[target]
        if b_max is not None and len(buf) >= b_max:
            active[buf.popleft()] = False
        g_feats[n_used], g_ident[n_used] = features[qi], target
        g_true[n_used], active[n_used], g_src[n_used] = labels[qi], True, int(qi)
        buf.append(n_used)
        n_used += 1
        appended += 1
        wrong += int(target != labels[qi])

    def flush(page_qs: List[int]) -> None:
        """The joint assignment for one page, applied after every query on it has been scored."""
        nonlocal abstained, linked_used
        if not page_qs:
            return
        idx = np.flatnonzero(active[:n_used])
        present = {int(c) for c in g_ident[idx]}
        identities = sorted(c for c in buffers if c in present)
        picks = assign_page(features, page_qs, link_groups, g_feats[idx], g_ident[idx],
                            identities, eta=eta, unlinked=unlinked, exclusive=exclusive)
        multi = {qi for grp in page_groups(page_qs, link_groups, unlinked) if len(grp) > 1
                 for qi in grp}
        for qi in page_qs:
            if qi in picks:
                commit(qi, picks[qi])
                linked_used += int(qi in multi)
            else:
                abstained += 1

    # Crops the stream has already delivered. The reciprocal policy may look only at these: P4 is
    # causal, so a mutual-neighbour test that consulted the rest of the chapter would be cheating.
    seen_idx: List[int] = list(sorted(seed_idx))
    pending: List[int] = []
    cur_page = None
    for qi in queries:
        if update_policy == "assign":
            page = page_ids[qi]
            if pending and page != cur_page:
                flush(pending)
                pending = []
            cur_page = page
        idx = np.flatnonzero(active[:n_used])
        gf, gl = g_feats[idx], g_ident[idx]
        m = compute_retrieval_metrics(features[qi][None], labels[qi][None], gf, gl)
        per_query.append(m)
        sims = gf @ features[qi]
        pred = int(gl[int(np.argmax(sims))])
        preds.append(pred); trues.append(int(labels[qi]))
        if update_policy == "frozen" or b_max == 0:
            continue
        if update_policy in GATED_POLICIES and _gate_score(update_policy, sims, gl, pred) < confidence_threshold:
            abstained += 1
            seen_idx.append(int(qi))
            continue
        if update_policy == "reciprocal" and not _is_reciprocal(
                features, seen_idx, int(qi), g_feats[idx[int(np.argmax(sims))]], reciprocal_k):
            abstained += 1
            seen_idx.append(int(qi))
            continue
        seen_idx.append(int(qi))
        if update_policy == "assign":
            pending.append(int(qi))
            continue
        if update_policy == "cluster":
            # Decide with the group, not the crop: average the query with the members of its link
            # group that the stream has already delivered, and match that. The group is right 90 %
            # of the time, so the average is a cleaner probe than the crop. Unlike the must-link
            # rule it needs no group member to be in the gallery already, which is what holds that
            # rule to 2.5 % of the stream.
            grp = link_groups[int(qi)]
            if grp is None:
                probe = features[qi]
            else:
                mates = [j for j in seen_idx if link_groups[j] == grp]
                probe = features[[int(qi)] + mates].mean(axis=0)
                probe = probe / max(1e-12, float(np.linalg.norm(probe)))
                if mates:
                    linked_used += 1
            target = int(gl[int(np.argmax(gf @ probe))])
        elif update_policy in LINK_POLICIES:
            linked = _linked_identity(link_groups, g_src, g_ident, active, n_used, int(qi))
            if linked is None:
                if update_policy == "mustlink":
                    abstained += 1
                    continue
                target = pred                       # mustlink_predicted falls back to the top-1
            else:
                target = linked
                linked_used += 1
        else:
            target = int(labels[qi]) if update_policy == "oracle" else pred
        commit(int(qi), target)
    if update_policy == "assign":
        flush(pending)

    scored = [m for m in per_query if m["n_queries"] == 1]
    out: Dict = {key: float(np.mean([m[key] for m in scored])) if scored else float("nan")
                 for key in ("mAP", "MRR", "R1", "R5", "R10")}
    out["n_queries"] = len(scored)
    out["n_skipped"] = len(per_query) - len(scored)
    preds_a, trues_a = np.asarray(preds), np.asarray(trues)
    correct = preds_a == trues_a
    out["R1_identity"] = float(correct.mean()) if len(correct) else float("nan")
    out["wrong_append"] = float(wrong / appended) if appended else 0.0
    grown = active[:n_used].copy(); grown[: len(seed_idx)] = False        # non-seed exemplars still present
    # index the same prefix the mask was cut from: the storage arrays are allocated for the worst
    # case, and a policy that declines to append leaves n_used short of it
    ident, true = g_ident[:n_used], g_true[:n_used]
    out["contamination"] = float((ident[grown] != true[grown]).mean()) if grown.any() else 0.0
    quart = np.array_split(np.arange(len(correct)), 4)
    out["r1_by_quartile"] = [float(correct[q].mean()) if len(q) else float("nan") for q in quart]
    out["n_appended"] = int(appended)
    out["n_abstained"] = int(abstained)
    out["confidence_threshold"] = confidence_threshold
    out["reciprocal_k"] = reciprocal_k
    out["n_linked"] = int(linked_used)
    out["append_rate"] = float(appended / len(queries)) if queries else 0.0
    out["n_gallery_final"] = int(active[:n_used].sum())
    out.update({"n_gallery_only_identities": len(gallery_only), "n_seeded_identities": len(seed_map),
                "n_gallery_initial": len(seed_idx), "k": int(k), "strategy": strategy, "seed": int(seed),
                "b_max": None if b_max is None else int(b_max), "update_policy": update_policy})
    if update_policy == "assign":
        out.update({"eta": float(eta), "unlinked": unlinked, "exclusive": bool(exclusive)})
    return out
