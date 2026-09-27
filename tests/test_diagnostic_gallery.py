"""Regression tests for the gallery semantics the diagnostics depend on.

Every gallery entry has two identities: the one it is *filed under* and its *true* one. Using one
where the other is needed is an easy mistake in either direction, and it changes a diagnostic's
numbers without raising anything. The tests below pin which identity each operation uses.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from _gallery import Gallery  # noqa: E402


def _toy():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(6, 4))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f, [0, 1, 2, 0, 1, 2]


def test_misfiled_crop_is_not_a_reference_for_its_filed_identity():
    """A misfiled crop is not a reference for the identity it is filed under. Append-by-top-1
    misfiles 78.8 % of 467 crops, and crediting each to the name it landed under would make the
    worst append policy look like the one that removes the most stream distance."""
    f, labels = _toy()
    g = Gallery(f, labels, seed_idx=[0, 1, 2])
    before = len(g.references_of(2))
    assert g.append(4, 2) is False          # crop 4 is identity 1, filed under 2
    assert len(g.references_of(2)) == before, "a misfiled crop must not count as a reference"


def test_misfiled_crop_is_not_a_reference_for_its_true_identity_either():
    """It cannot answer for its true identity: a match on it returns the name it is filed under."""
    f, labels = _toy()
    g = Gallery(f, labels, seed_idx=[0, 1, 2])
    before = len(g.references_of(1))
    g.append(4, 2)                           # crop 4 truly is 1
    assert len(g.references_of(1)) == before


def test_correct_append_is_a_reference():
    f, labels = _toy()
    g = Gallery(f, labels, seed_idx=[0, 1, 2])
    assert g.append(3, 0) is True
    assert len(g.references_of(0)) == 2


def test_capture_precision_uses_the_filed_identity():
    """Capture precision (`p_eff`) uses the filed identity: an exemplar that genuinely is the query's
    character but sits under another name still returns a wrong answer. Scoring it as a success
    would inflate p_eff from about 0.25 to about 0.55."""
    f, labels = _toy()
    g = Gallery(f, labels, seed_idx=[0])
    # file crop 3 (identity 0) under identity 1, then query with crop 3 itself
    g.append(3, 1)
    pred, grown = g.rank(3)
    assert grown is True, "the nearest entry is the grown one"
    assert pred == 1, "the prediction is the FILED identity, not the true one"
    assert pred != labels[3], "so this capture is a failure, however similar the crops are"


def test_seeds_are_protected_from_eviction():
    f, labels = _toy()
    g = Gallery(f, labels, seed_idx=[0], b_max=1)
    g.append(3, 0)
    g.append(1, 0)
    refs = g.references_of(0)
    assert len(refs) == 1, "b_max bounds growth, and the seed is not in the growth buffer"


def test_static_ranking_ignores_grown_entries():
    """a+ is what the frozen gallery scored on the captured queries, so growth must not leak in."""
    f, labels = _toy()
    g = Gallery(f, labels, seed_idx=[0, 1, 2])
    before = g.rank_static(5)
    for _ in range(3):
        g.append(3, 2)
    assert g.rank_static(5) == before, "rank_static must not see appended rows"
