"""Page-level joint assignment: the append decision made for a whole page at once.

Four append rules have been measured on the test set and all four decide one crop at a time.
Confidence is dead, margin works on one backbone of three, reciprocity works on none, and the
must-link rule is the only one whose appends clear the 40.9 % break-even precision (at 90.6 %),
but it fires on 2.5 % of the stream, because it needs a member of the query's own page group to be
in the gallery already. That is the recall wall, and it is a property of the *unit of decision*
rather than of the score: a page whose characters are all new to the gallery can never be anchored
by a rule that only copies labels it can already see.

MagiV2's own chapter-wide inference does not have that problem, and the reason is in its
formulation rather than its features. It assigns *page clusters* to a character bank jointly, under
three constraints: the crops of one page cluster take one identity (must-link, an equality), two
clusters on the same page take different identities (cannot-link, an exclusion), and any cluster
may take a reject column priced at a constant eta. This module is that formulation with the
character bank replaced by the P4 gallery, so the assignment is what grows the gallery and the
gallery is what the next page is assigned against. Anchoring then propagates across pages: a page
needs one confident cluster, not a seed.

Collapsing each must-link group to a single node makes the integer program unnecessary. With the
groups collapsed, every constraint that remains is "distinct nodes take distinct columns", so the
problem is a rectangular linear sum assignment and `scipy.optimize.linear_sum_assignment` solves it
exactly. Rejection is a private column per group, priced at its member count times eta, which is
what pricing every member's reject at eta comes to in MagiV2's crop-level objective.

Distances mirror theirs: Euclidean between unit-norm embeddings (so eta = 0.75 is their default,
equivalent to a cosine of 0.719), a group's cost is the sum over its members, and an identity's
distance is to its nearest live exemplar.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

UNLINKED_MODES = ("abstain", "free")
ETA_DEFAULT = 0.75


def page_groups(page_queries: Sequence[int], link_groups: Sequence,
                unlinked: str = "abstain") -> List[List[int]]:
    """The page's crops collapsed into must-link groups, in first-appearance order.

    A crop MagiV2's detector never matched has no group. Under "abstain" it is dropped, which is the
    precision-first reading: the evidence that put the other crops in a group is exactly the evidence
    it lacks. Under "free" it becomes a singleton group and competes for an identity like any other,
    which also makes it cannot-link with the rest of the page.
    """
    if unlinked not in UNLINKED_MODES:
        raise ValueError(f"unlinked must be one of {UNLINKED_MODES}, got {unlinked!r}")
    out: List[List[int]] = []
    by_key: Dict[object, int] = {}
    for qi in page_queries:
        key = link_groups[qi]
        if key is None:
            if unlinked == "abstain":
                continue
            out.append([int(qi)])
            continue
        if key not in by_key:
            by_key[key] = len(out)
            out.append([])
        out[by_key[key]].append(int(qi))
    return out


def group_costs(features: np.ndarray, groups: Sequence[Sequence[int]],
                g_feats: np.ndarray, g_ident: np.ndarray,
                identities: Sequence[int]) -> np.ndarray:
    """Euclidean cost of filing each group under each identity, summed over the group's members.

    The distance to an identity is the distance to its nearest live exemplar, which is the same
    quantity the top-1 rule ranks by, so the two rules are compared on their decision structure
    and not on a change of score.
    """
    if not groups or not len(identities):
        return np.zeros((len(groups), len(identities)))
    members = [np.asarray(g, dtype=int) for g in groups]
    sims = features[np.concatenate(members)] @ g_feats.T             # (n_members, n_gallery)
    best = np.empty((len(members), len(identities)))
    at = 0
    for gi, m in enumerate(members):
        block = sims[at:at + len(m)]
        at += len(m)
        for ci, c in enumerate(identities):
            col = block[:, g_ident == c]
            best[gi, ci] = float(np.sqrt(np.maximum(0.0, 2.0 - 2.0 * col.max(axis=1))).sum())
    return best


def solve(costs: np.ndarray, sizes: Sequence[int], eta: float,
          exclusive: bool = True) -> List[Optional[int]]:
    """Assign each group a distinct identity column or its own reject column.

    Returns one entry per group: the column index into `identities`, or None for reject.
    `exclusive=False` drops the cannot-link exclusion, so every group takes its own nearest
    identity independently. This is the control that separates what the joint solve is worth from
    what the reject column and the must-link grouping are worth.
    """
    n_g = costs.shape[0]
    n_c = costs.shape[1]
    if n_g == 0:
        return []
    if not exclusive:
        floor = eta * np.asarray(sizes, dtype=float)
        pick = costs.argmin(axis=1)
        return [int(c) if costs[g, c] < floor[g] else None for g, c in enumerate(pick)]
    # One reject column per group. Its price is capped just above the cost of assigning every
    # group, so eta = inf means "never reject unless there are too few identities to go round"
    # and the program stays feasible; foreign reject columns are priced out of reach.
    scale = float(costs.max()) if costs.size else 0.0
    cap = scale * n_g + 1.0
    diag = np.minimum(eta * np.asarray(sizes, dtype=float), cap)
    reject = np.full((n_g, n_g), (cap + scale) * (n_g + 2) + 1.0)
    np.fill_diagonal(reject, diag)
    full = np.concatenate([costs, reject], axis=1)
    rows, cols = linear_sum_assignment(full)
    out: List[Optional[int]] = [None] * n_g
    for r, c in zip(rows, cols):
        out[int(r)] = int(c) if c < n_c else None
    return out


def assign_page(features: np.ndarray, page_queries: Sequence[int], link_groups: Sequence,
                g_feats: np.ndarray, g_ident: np.ndarray, identities: Sequence[int],
                eta: float = ETA_DEFAULT, unlinked: str = "abstain",
                exclusive: bool = True) -> Dict[int, int]:
    """Crop index -> identity, for the crops this page's joint assignment commits to."""
    groups = page_groups(page_queries, link_groups, unlinked)
    if not groups or not len(identities):
        return {}
    costs = group_costs(features, groups, g_feats, g_ident, identities)
    picks = solve(costs, [len(g) for g in groups], eta, exclusive=exclusive)
    out: Dict[int, int] = {}
    for members, col in zip(groups, picks):
        if col is None:
            continue
        for qi in members:
            out[int(qi)] = int(identities[col])
    return out
