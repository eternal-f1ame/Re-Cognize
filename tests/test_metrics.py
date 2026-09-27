"""Tests for recognize.metrics. Values are hand-computed."""
from __future__ import annotations

import numpy as np
import pytest


def _unit(rows):
    a = np.array(rows, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


@pytest.fixture
def toy():
    # gallery: one-hot basis; labels A=0 (g1,g3), B=1 (g2), C=2 (g4)
    g = np.eye(4)
    gl = np.array([0, 1, 0, 2])
    # q1 (A): ranks g1, g3 first -> AP 1.0 ; q2 (B): g1 then g2 -> AP 0.5 ; q3 (A): g1, g2, g3 -> AP (1 + 2/3)/2
    q = _unit([[0.9, 0.1, 0.8, 0.0], [0.9, 0.8, 0.0, 0.0], [0.9, 0.8, 0.7, 0.0]])
    ql = np.array([0, 1, 0])
    return q, ql, g, gl


class TestRetrievalMetrics:
    def test_hand_computed_map_mrr_cmc(self, toy):
        from recognize.metrics import compute_retrieval_metrics
        m = compute_retrieval_metrics(*toy)
        assert m["mAP"] == pytest.approx((1.0 + 0.5 + (1 + 2 / 3) / 2) / 3, abs=1e-9)
        assert m["MRR"] == pytest.approx((1.0 + 0.5 + 1.0) / 3, abs=1e-9)
        assert m["R1"] == pytest.approx(2 / 3) and m["R5"] == pytest.approx(1.0) and m["R10"] == pytest.approx(1.0)
        assert m["n_queries"] == 3 and m["n_skipped"] == 0

    def test_query_without_positive_is_skipped_and_counted(self, toy):
        from recognize.metrics import compute_retrieval_metrics
        q, ql, g, gl = toy
        q2 = np.vstack([q, _unit([[0.0, 0.0, 0.0, 1.0]])]); ql2 = np.append(ql, 9)
        m = compute_retrieval_metrics(q2, ql2, g, gl)
        assert m["n_queries"] == 3 and m["n_skipped"] == 1
        assert m["mAP"] == pytest.approx((1.0 + 0.5 + (1 + 2 / 3) / 2) / 3, abs=1e-9)

    def test_ranks_clipped_to_gallery_size(self, toy):
        from recognize.metrics import compute_retrieval_metrics
        m = compute_retrieval_metrics(*toy, ranks=(1, 50))
        assert m["R50"] == pytest.approx(1.0)

    def test_all_queries_skipped_gives_nan_not_crash(self, toy):
        from recognize.metrics import compute_retrieval_metrics
        q, ql, g, gl = toy
        m = compute_retrieval_metrics(q, np.array([7, 7, 7]), g, gl)
        assert m["n_queries"] == 0 and np.isnan(m["mAP"])


class TestClusteringMetrics:
    def test_perfect_permuted_partition(self):
        from recognize.metrics import clustering_metrics
        m = clustering_metrics(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
        assert m["clusters"] == 2 and m["purity"] == 1.0 and m["ari"] == pytest.approx(1.0)
        assert m["nmi"] == pytest.approx(1.0) and m["hungarian"] == pytest.approx(1.0)

    def test_singleton_partition(self):
        from recognize.metrics import clustering_metrics
        m = clustering_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 2, 3]))
        assert m["clusters"] == 4 and m["purity"] == 1.0
        assert m["ari"] == pytest.approx(0.0) and m["hungarian"] == pytest.approx(0.5)

    def test_one_cluster(self):
        from recognize.metrics import clustering_metrics
        m = clustering_metrics(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]))
        assert m["clusters"] == 1 and m["purity"] == pytest.approx(0.75) and m["hungarian"] == pytest.approx(0.75)


class TestPrecomputedSimilarities:
    """A diagnostic may rank by something other than one dot product, without forking the metric."""

    def _data(self, seed=0):
        rng = np.random.default_rng(seed)
        q = rng.normal(size=(12, 8)); g = rng.normal(size=(20, 8))
        q /= np.linalg.norm(q, axis=1, keepdims=True); g /= np.linalg.norm(g, axis=1, keepdims=True)
        return q, rng.integers(0, 4, 12), g, rng.integers(0, 4, 20)

    def test_passing_the_dot_product_reproduces_the_feature_path_exactly(self):
        from recognize.metrics import compute_retrieval_metrics
        q, ql, g, gl = self._data()
        a = compute_retrieval_metrics(q, ql, g, gl)
        b = compute_retrieval_metrics(q, ql, g, gl, sims=q @ g.T)
        assert a == b

    def test_a_fused_score_changes_the_ranking(self):
        from recognize.metrics import compute_retrieval_metrics
        q, ql, g, gl = self._data()
        rng = np.random.default_rng(1)
        other = rng.normal(size=(12, 8)); other /= np.linalg.norm(other, axis=1, keepdims=True)
        fused = 0.5 * (q @ g.T) + 0.5 * (other @ g.T)
        assert compute_retrieval_metrics(q, ql, g, gl, sims=fused)["mAP"] != \
               compute_retrieval_metrics(q, ql, g, gl)["mAP"]

    def test_a_mis_shaped_matrix_is_refused_rather_than_broadcast(self):
        from recognize.metrics import compute_retrieval_metrics
        q, ql, g, gl = self._data()
        with pytest.raises(ValueError, match="expected"):
            compute_retrieval_metrics(q, ql, g, gl, sims=np.zeros((3, 3)))
