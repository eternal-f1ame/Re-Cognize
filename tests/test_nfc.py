"""Neighbour Feature Centralization: the reciprocity rule and the aggregation, on hand-made cases.

The module under test is analysis/nfc.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DIAG = REPO / "analysis"
if str(DIAG) not in sys.path:
    sys.path.insert(0, str(DIAG))


def unit(rows):
    a = np.asarray(rows, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


class TestMutualNeighbours:
    def test_a_reciprocated_pair_is_kept(self):
        import nfc
        f = unit([[1, 0], [0.99, 0.14], [0, 1]])      # 0 and 1 are each other's nearest
        m = nfc.mutual_neighbours(f, k1=1, k2=1)
        assert list(m[0]) == [1] and list(m[1]) == [0]

    def test_an_unreciprocated_neighbour_is_dropped(self):
        import nfc
        # 2 points at 0 deg and 1 deg apart, a third far away whose nearest is 0 but which is
        # nobody's nearest: with k1=k2=1 the third keeps nothing and nobody keeps it
        f = unit([[1, 0], [0.9998, 0.0175], [0.7, 0.7]])
        m = nfc.mutual_neighbours(f, k1=1, k2=1)
        assert list(m[2]) == []                        # 2's nearest is 1, but 1's nearest is 0
        assert 2 not in list(m[0]) and 2 not in list(m[1])

    def test_a_larger_k2_readmits_it(self):
        import nfc
        f = unit([[1, 0], [0.9998, 0.0175], [0.7, 0.7]])
        assert list(nfc.mutual_neighbours(f, k1=1, k2=2)[2]) == [1]

    def test_self_is_never_a_neighbour(self):
        import nfc
        f = unit(np.eye(5) + 0.01)
        assert all(i not in list(nb) for i, nb in enumerate(nfc.mutual_neighbours(f, 3, 3)))

    def test_k1_bounds_the_candidate_set(self):
        import nfc
        rng = np.random.default_rng(0)
        f = unit(rng.normal(size=(30, 8)))
        assert all(len(nb) <= 2 for nb in nfc.mutual_neighbours(f, k1=2, k2=30))


class TestCentralize:
    def test_it_sums_the_mutual_neighbours_and_renormalises(self):
        import nfc
        f = unit([[1, 0], [0.99, 0.14], [0, 1]])
        out = nfc.centralize(f, k1=1, k2=1)
        expect = f[0] + f[1]
        assert np.allclose(out[0], expect / np.linalg.norm(expect))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_a_point_with_no_mutual_neighbour_is_unchanged(self):
        import nfc
        f = unit([[1, 0], [0.9998, 0.0175], [0.7, 0.7]])
        out = nfc.centralize(f, k1=1, k2=1)
        assert np.allclose(out[2], f[2])

    def test_it_never_touches_the_input(self):
        import nfc
        rng = np.random.default_rng(1)
        f = unit(rng.normal(size=(20, 6)))
        before = f.copy()
        nfc.centralize(f, 4, 2)
        assert np.array_equal(f, before)


class TestNeighbourPurity:
    def test_a_clean_neighbourhood_scores_one(self):
        import nfc
        f = unit([[1, 0], [0.99, 0.14], [0, 1], [0.14, 0.99]])
        labels = np.array([0, 0, 1, 1])
        p = nfc.neighbour_purity(f, labels, k1=1, k2=1)
        assert p["purity"] == pytest.approx(1.0) and p["empty_frac"] == 0.0

    def test_a_wrong_neighbour_is_counted(self):
        import nfc
        f = unit([[1, 0], [0.99, 0.14], [0, 1], [0.14, 0.99]])
        labels = np.array([0, 1, 2, 3])                # every neighbour is now wrong
        assert nfc.neighbour_purity(f, labels, k1=1, k2=1)["purity"] == pytest.approx(0.0)

    def test_empty_neighbourhoods_are_reported_not_averaged_in(self):
        import nfc
        f = unit([[1, 0], [0.9998, 0.0175], [0.7, 0.7]])
        p = nfc.neighbour_purity(f, np.array([0, 0, 1]), k1=1, k2=1)
        assert p["empty_frac"] == pytest.approx(1 / 3) and p["purity"] == pytest.approx(1.0)


class TestSideBlending:
    def test_query_only_leaves_the_gallery_rows_alone(self):
        import nfc
        plain = unit(np.arange(12, dtype=float).reshape(6, 2) + 1)
        central = plain[::-1].copy()
        out = nfc.blend(plain, central, [0, 2], "query")
        assert np.array_equal(out[0], central[0]) and np.array_equal(out[2], central[2])
        assert np.array_equal(out[1], plain[1]) and np.array_equal(out[3], plain[3])

    def test_both_replaces_everything(self):
        import nfc
        plain = unit(np.arange(12, dtype=float).reshape(6, 2) + 1)
        central = plain[::-1].copy()
        assert np.array_equal(nfc.blend(plain, central, [0], "both"), central)

    def test_blending_does_not_mutate_the_inputs(self):
        import nfc
        plain = unit(np.arange(12, dtype=float).reshape(6, 2) + 1)
        central = plain[::-1].copy()
        a, b = plain.copy(), central.copy()
        nfc.blend(plain, central, [1, 4], "gallery")
        assert np.array_equal(plain, a) and np.array_equal(central, b)
