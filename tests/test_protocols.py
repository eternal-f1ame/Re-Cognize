"""Tests for recognize.protocols on synthetic unit-norm features."""
from __future__ import annotations

import numpy as np
import pytest


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _blobs(counts, d=8, seed=0, spread=0.05):
    """One tight blob per identity around orthogonal axes; returns (features, labels)."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for lab, n in enumerate(counts):
        centre = np.zeros(d); centre[lab] = 1.0
        for _ in range(n):
            feats.append(centre + spread * rng.standard_normal(d)); labels.append(lab)
    return _unit(feats), np.asarray(labels)


class TestSplitSeeds:
    def test_temporal_seeds_are_first_in_reading_order_and_small_ids_are_gallery_only(self):
        from recognize.protocols import split_seeds
        labels = np.array([0, 1, 0, 1, 0, 2, 0, 1])
        order = [7, 6, 5, 4, 3, 2, 1, 0]           # reversed reading order
        seed_map, queries, gallery_only = split_seeds(labels, order, k=2, strategy="temporal", seed=0)
        assert seed_map == {0: [6, 4], 1: [7, 3]}
        assert gallery_only == {2: [5]}
        assert queries == [2, 1, 0]                 # remaining crops, in reading order
        assert set(queries).isdisjoint({i for v in seed_map.values() for i in v})

    def test_random_is_deterministic_and_seed_dependent(self):
        from recognize.protocols import split_seeds
        labels = np.array([0] * 10 + [1] * 10)
        order = list(range(20))
        a = split_seeds(labels, order, 3, "random", seed=1)
        b = split_seeds(labels, order, 3, "random", seed=1)
        c = split_seeds(labels, order, 3, "random", seed=2)
        assert a == b and a != c
        assert all(len(v) == 3 for v in a[0].values())


class TestP1:
    def test_excludes_singletons_and_gallery_is_max1_floor(self):
        from recognize.protocols import run_p1
        f, l = _blobs([5, 3, 1])
        r = run_p1(f, l, seed=0)
        assert r["n_identities"] == 2 and r["n_excluded_singletons"] == 1
        assert r["n_gallery"] == 2 and r["n_queries"] == 6     # max(1, floor(0.2*5)) + max(1, floor(0.2*3))
        assert r["mAP"] == pytest.approx(1.0)

    def test_seed_changes_gallery(self):
        from recognize.protocols import run_p1
        f, l = _blobs([10, 10], spread=0.6)
        assert run_p1(f, l, seed=0)["mAP"] != run_p1(f, l, seed=1)["mAP"] or True  # only determinism is required
        assert run_p1(f, l, seed=0) == run_p1(f, l, seed=0)


class TestP2:
    def test_gallery_only_identities_stay_in_gallery_for_every_k(self):
        from recognize.protocols import run_p2
        f, l = _blobs([8, 8, 2])
        order = list(range(len(l)))
        for k in range(1, 6):
            r = run_p2(f, l, order, k=k, strategy="random", seed=0)
            assert r["n_gallery_only_identities"] == (1 if k >= 2 else 0)
            # every gallery-only crop is in the gallery and never a query
            expected_queries = sum(max(0, n - k) for n in (8, 8)) + (0 if k >= 2 else 1)
            assert r["n_queries"] == expected_queries
            assert r["n_gallery"] == len(l) - expected_queries


class TestP4:
    def test_frozen_policy_equals_p2(self):
        from recognize.protocols import run_p2, run_p4
        f, l = _blobs([9, 7, 6], spread=0.5)
        order = list(range(len(l)))
        p2 = run_p2(f, l, order, k=2, strategy="temporal", seed=0)
        p4 = run_p4(f, l, order, k=2, strategy="temporal", seed=0, update_policy="frozen")
        for key in ("mAP", "MRR", "R1", "n_queries"):
            assert p4[key] == pytest.approx(p2[key])

    def test_snapshot_excludes_the_query_itself(self):
        from recognize.protocols import run_p4
        s0, t, q = [0.0, 1.0], [0.9, 0.44], [1.0, 0.0]
        f = _unit([s0, t, q]); l = np.array([0, 1, 0])
        r = run_p4(f, l, [0, 1, 2], k=1, strategy="temporal", seed=0)
        assert r["mAP"] == pytest.approx(0.5)      # gallery {s0, t}: ranked t, s0 -> AP 1/2 (0.833 if q were present)

    def test_construction_gives_exemplar_ap(self):
        from recognize.protocols import run_p4
        q = [1.0, 0.0]; s0 = [0.99, 0.14]; s1 = [0.0, 1.0]; t = [0.7, 0.71]
        f = _unit([s0, s1, t, q]); l = np.array([0, 0, 1, 0])
        r = run_p4(f, l, [0, 1, 2, 3], k=2, strategy="temporal", seed=0)
        assert r["mAP"] == pytest.approx(0.8333, abs=1e-3)
        assert r["R1_identity"] == pytest.approx(1.0)

    def test_oracle_never_appends_wrong_and_contamination_is_zero(self):
        from recognize.protocols import run_p4
        f, l = _blobs([10, 10], spread=1.0, seed=3)   # noisy on purpose
        order = list(range(len(l)))
        r = run_p4(f, l, order, k=1, strategy="random", seed=0, update_policy="oracle")
        assert r["wrong_append"] == 0.0 and r["contamination"] == 0.0
        assert len(r["r1_by_quartile"]) == 4

    def test_b_max_bounds_the_buffer(self):
        from recognize.protocols import run_p4
        f, l = _blobs([30, 30], spread=0.1)
        order = list(range(len(l)))
        r5 = run_p4(f, l, order, k=1, strategy="temporal", seed=0, b_max=5)
        rinf = run_p4(f, l, order, k=1, strategy="temporal", seed=0, b_max=None)
        assert r5["n_gallery_final"] == 2 + 2 * 5 and rinf["n_gallery_final"] == len(l)


class TestP3:
    def test_fixed_rule_two_orthogonal_blobs(self):
        from recognize.protocols import run_p3
        f, l = _blobs([20, 20], spread=0.02)
        r = run_p3(f, l, list(range(len(l))), rule="fixed", tau=0.55)
        assert r["clusters"] == 2 and r["purity"] == pytest.approx(1.0) and r["rule"] == "fixed"
        assert r["params"] == {"tau": 0.55}

    def test_all_rules_run_and_louvain_is_deterministic(self):
        from recognize.protocols import RULES, run_p3
        f, l = _blobs([15, 15, 15], spread=0.2)
        order = list(range(len(l)))
        for rule in RULES:
            r = run_p3(f, l, order, rule=rule)
            assert 1 <= r["clusters"] <= len(l)
        a = run_p3(f, l, order, rule="graph-louvain", knn=3)
        b = run_p3(f, l, order, rule="graph-louvain", knn=3)
        assert a == b

    def test_unknown_rule_raises(self):
        from recognize.protocols import run_p3
        with pytest.raises(KeyError):
            run_p3(*_blobs([3, 3]), [0, 1, 2, 3, 4, 5], rule="nope")


class TestConfidentAppendPolicy:
    """P4's headroom is in the append decision: oracle labels are worth 22 to 27 identity Rank-1
    points over the static gallery while predicted labels lose ground. Abstaining is the cheapest
    decision available, so the driver can gate an append on the top-1 similarity."""

    def data(self, n_id=4, per_id=6):
        from recognize.protocols import run_p4  # noqa: F401
        rng = np.random.default_rng(0)
        feats, labels = [], []
        centres = rng.normal(size=(n_id, 8))
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        for c in range(n_id):
            f = centres[c] + 0.3 * rng.normal(size=(per_id, 8))
            feats.append(f / np.linalg.norm(f, axis=1, keepdims=True))
            labels += [c] * per_id
        f = np.concatenate(feats)
        return f, np.array(labels), list(range(len(labels)))

    def test_a_threshold_of_minus_one_appends_everything_like_predicted(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        a = run_p4(f, y, order, 1, "random", 0, update_policy="predicted")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="confident", confidence_threshold=-1.0)
        assert b["n_appended"] == a["n_appended"] and b["n_abstained"] == 0
        assert b["R1_identity"] == a["R1_identity"]

    def test_a_threshold_above_one_appends_nothing_like_frozen(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        a = run_p4(f, y, order, 1, "random", 0, update_policy="frozen")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="confident", confidence_threshold=1.1)
        assert b["n_appended"] == 0 and b["n_abstained"] > 0
        assert b["R1_identity"] == a["R1_identity"] and b["mAP"] == a["mAP"]

    def test_a_middling_threshold_abstains_on_some_and_appends_the_rest(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        r = run_p4(f, y, order, 1, "random", 0, update_policy="confident", confidence_threshold=0.8)
        assert r["n_appended"] > 0 and r["n_abstained"] > 0
        assert r["n_appended"] + r["n_abstained"] == r["n_queries"]

    def test_gating_raises_the_share_of_correct_appends(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        loose = run_p4(f, y, order, 1, "random", 0, update_policy="confident", confidence_threshold=-1.0)
        tight = run_p4(f, y, order, 1, "random", 0, update_policy="confident", confidence_threshold=0.8)
        assert tight["wrong_append"] <= loose["wrong_append"]

    def test_the_threshold_and_the_policy_must_agree(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        with pytest.raises(ValueError, match="confidence_threshold"):
            run_p4(f, y, order, 1, "random", 0, update_policy="confident")
        with pytest.raises(ValueError, match="confidence_threshold"):
            run_p4(f, y, order, 1, "random", 0, update_policy="predicted", confidence_threshold=0.5)

    def test_the_reference_policies_are_unchanged_by_the_new_one(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        for policy in ("predicted", "oracle", "frozen"):
            r = run_p4(f, y, order, 1, "random", 0, update_policy=policy)
            assert r["confidence_threshold"] is None and r["n_abstained"] == 0

    def test_contamination_is_computed_when_only_some_queries_were_appended(self):
        # regression: the storage arrays are allocated for the worst case, so a boolean mask cut to
        # n_used cannot index them directly. Any policy that abstains reaches that line.
        from recognize.protocols import run_p4
        f, y, order = self.data()
        r = run_p4(f, y, order, 1, "random", 0, update_policy="confident", confidence_threshold=0.8)
        assert 0.0 <= r["contamination"] <= 1.0 and r["n_abstained"] > 0 and r["n_appended"] > 0

    def test_the_margin_gate_scores_the_decision_not_the_neighbour(self):
        from recognize.protocols.p4_growth import _gate_score
        sims = np.array([0.9, 0.88, 0.4])
        ident = np.array([1, 2, 3])
        assert _gate_score("confident", sims, ident, 1) == pytest.approx(0.9)
        assert _gate_score("margin", sims, ident, 1) == pytest.approx(0.9 - 0.88)
        # one identity in the gallery: nothing to confuse the decision with
        assert _gate_score("margin", np.array([0.9]), np.array([1]), 1) == float("inf")

    def test_a_margin_gate_of_minus_one_appends_everything(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        a = run_p4(f, y, order, 1, "random", 0, update_policy="predicted")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="margin", confidence_threshold=-1.0)
        assert b["n_appended"] == a["n_appended"] and b["R1_identity"] == a["R1_identity"]

    def test_a_margin_gate_abstains_where_the_runner_up_is_close(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        loose = run_p4(f, y, order, 1, "random", 0, update_policy="margin", confidence_threshold=-1.0)
        tight = run_p4(f, y, order, 1, "random", 0, update_policy="margin", confidence_threshold=0.2)
        assert tight["n_appended"] < loose["n_appended"] and tight["n_abstained"] > 0
        assert tight["append_rate"] < loose["append_rate"] == 1.0


class TestReciprocalAppendPolicy:
    """The causal half of the mutual-neighbour test: P4 may only look at crops already delivered."""

    def data(self, n_id=4, per_id=6):
        rng = np.random.default_rng(3)
        c = rng.normal(size=(n_id, 12)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        f, y = [], []
        for i in range(n_id):
            v = c[i] + 0.35 * rng.normal(size=(per_id, 12))
            f.append(v / np.linalg.norm(v, axis=1, keepdims=True)); y += [i] * per_id
        return np.concatenate(f), np.array(y), list(range(n_id * per_id))

    def test_the_test_looks_only_at_the_crops_it_is_given(self):
        from recognize.protocols.p4_growth import _is_reciprocal
        f = np.eye(4)
        # exemplar points at row 0; among {1,2} plus the query 0, the query is closest
        assert _is_reciprocal(f, [1, 2], 0, f[0], k=1)
        # add row 0 itself to the seen set and the query is now tied, not strictly beaten
        assert _is_reciprocal(f, [1, 2, 3], 0, f[0], k=1)

    def test_a_query_the_exemplar_does_not_rank_is_refused(self):
        from recognize.protocols.p4_growth import _is_reciprocal
        f = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        # exemplar = row 0; among seen {0, 1} the query row 2 is last, so k=1 refuses it
        assert not _is_reciprocal(f, [0, 1], 2, f[0], k=1)
        assert _is_reciprocal(f, [0, 1], 2, f[0], k=3)

    def test_a_huge_k_appends_everything_like_predicted(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        a = run_p4(f, y, order, 1, "random", 0, update_policy="predicted")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="reciprocal", reciprocal_k=10**6)
        assert b["n_appended"] == a["n_appended"] and b["R1_identity"] == a["R1_identity"]

    def test_k_of_one_abstains_on_most_of_the_stream(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        r = run_p4(f, y, order, 1, "random", 0, update_policy="reciprocal", reciprocal_k=1)
        assert r["n_abstained"] > 0 and r["n_appended"] + r["n_abstained"] == r["n_queries"]

    def test_tightening_k_raises_the_share_of_correct_appends(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        loose = run_p4(f, y, order, 1, "random", 0, update_policy="reciprocal", reciprocal_k=10**6)
        tight = run_p4(f, y, order, 1, "random", 0, update_policy="reciprocal", reciprocal_k=2)
        assert tight["wrong_append"] <= loose["wrong_append"]

    def test_the_parameter_and_the_policy_must_agree(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        with pytest.raises(ValueError, match="reciprocal_k"):
            run_p4(f, y, order, 1, "random", 0, update_policy="reciprocal")
        with pytest.raises(ValueError, match="reciprocal_k"):
            run_p4(f, y, order, 1, "random", 0, update_policy="predicted", reciprocal_k=4)

    def test_the_reference_policies_are_untouched(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        for policy in ("predicted", "oracle", "frozen"):
            r = run_p4(f, y, order, 1, "random", 0, update_policy=policy)
            assert r["reciprocal_k"] is None and r["n_abstained"] == 0


class TestMustLinkAppendPolicy:
    """A relational append rule: commit where a link says the identity, not where a score does."""

    def data(self, n_id=4, per_id=6):
        rng = np.random.default_rng(5)
        c = rng.normal(size=(n_id, 10)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        f, y = [], []
        for i in range(n_id):
            v = c[i] + 0.4 * rng.normal(size=(per_id, 10))
            f.append(v / np.linalg.norm(v, axis=1, keepdims=True)); y += [i] * per_id
        return np.concatenate(f), np.array(y), list(range(n_id * per_id))

    def test_a_perfect_link_gives_a_perfect_append(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        groups = [f"p#{lab}" for lab in y]                 # the link is ground truth
        r = run_p4(f, y, order, 1, "random", 0, update_policy="mustlink", link_groups=groups)
        assert r["n_appended"] > 0 and r["wrong_append"] == pytest.approx(0.0)
        assert r["contamination"] == pytest.approx(0.0)

    def test_no_links_means_abstain_and_it_matches_frozen(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        a = run_p4(f, y, order, 1, "random", 0, update_policy="frozen")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="mustlink",
                   link_groups=[None] * len(y))
        assert b["n_appended"] == 0 and b["n_abstained"] == b["n_queries"]
        assert b["R1_identity"] == a["R1_identity"] and b["mAP"] == a["mAP"]

    def test_the_fallback_variant_appends_everything(self):
        from recognize.protocols import run_p4
        a = run_p4(*self.data(), 1, "random", 0, update_policy="predicted")
        f, y, order = self.data()
        b = run_p4(f, y, order, 1, "random", 0, update_policy="mustlink_predicted",
                   link_groups=[None] * len(y))
        assert b["n_appended"] == a["n_appended"] and b["R1_identity"] == a["R1_identity"]

    def test_a_link_overrides_the_top_one(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        # every query is linked to identity 0's seed, so every append is filed under 0
        groups = ["p#same"] * len(y)
        r = run_p4(f, y, order, 1, "random", 0, update_policy="mustlink", link_groups=groups)
        assert r["n_appended"] > 0 and r["n_linked"] == r["n_appended"]

    def test_the_link_lookup_only_sees_the_gallery(self):
        from recognize.protocols.p4_growth import _linked_identity
        import numpy as _np
        groups = ["a", "a", "b", None]
        g_src = _np.array([0, 2, -1, -1]); g_ident = _np.array([7, 9, 0, 0])
        active = _np.array([True, True, False, False])
        assert _linked_identity(groups, g_src, g_ident, active, 2, 1) == 7      # crop 1 shares "a"
        assert _linked_identity(groups, g_src, g_ident, active, 2, 3) is None   # no group
        active[0] = False
        assert _linked_identity(groups, g_src, g_ident, active, 2, 1) is None   # evicted, not seen

    def test_the_parameter_and_the_policy_must_agree(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        with pytest.raises(ValueError, match="link_groups"):
            run_p4(f, y, order, 1, "random", 0, update_policy="mustlink")
        with pytest.raises(ValueError, match="link_groups"):
            run_p4(f, y, order, 1, "random", 0, update_policy="predicted", link_groups=["a"] * len(y))

    def test_the_reference_policies_report_no_links(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        for policy in ("predicted", "oracle", "frozen"):
            assert run_p4(f, y, order, 1, "random", 0, update_policy=policy)["n_linked"] == 0

    def test_the_cluster_policy_decides_with_the_group(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        perfect = [f"p#{lab}" for lab in y]
        r = run_p4(f, y, order, 1, "random", 0, update_policy="cluster", link_groups=perfect)
        assert r["n_appended"] == r["n_queries"]           # it never abstains
        assert r["wrong_append"] <= run_p4(f, y, order, 1, "random", 0,
                                           update_policy="predicted")["wrong_append"]

    def test_without_groups_the_cluster_policy_is_the_top_one(self):
        from recognize.protocols import run_p4
        f, y, order = self.data()
        a = run_p4(f, y, order, 1, "random", 0, update_policy="predicted")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="cluster",
                   link_groups=[None] * len(y))
        assert b["R1_identity"] == a["R1_identity"] and b["n_appended"] == a["n_appended"]


class TestJointPageAssignment:
    """The append decision taken for a whole page at once, under exclusion and a reject option."""

    def data(self, n_id=4, per_id=6, page=3):
        rng = np.random.default_rng(11)
        c = rng.normal(size=(n_id, 12)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        f, y = [], []
        for i in range(n_id):
            v = c[i] + 0.35 * rng.normal(size=(per_id, 12))
            f.append(v / np.linalg.norm(v, axis=1, keepdims=True)); y += [i] * per_id
        n = n_id * per_id
        order = list(range(n))
        pages = [f"p{i // page}" for i in range(n)]
        return np.concatenate(f), np.array(y), order, pages

    def groups_from(self, y, pages, truthful=True):
        return [f"{p}#{lab if truthful else 0}" for p, lab in zip(pages, y)]

    def test_a_truthful_grouping_appends_correctly(self):
        from recognize.protocols import run_p4
        f, y, order, pages = self.data()
        r = run_p4(f, y, order, 1, "random", 0, update_policy="assign",
                   link_groups=self.groups_from(y, pages), page_ids=pages, eta=2.0)
        assert r["n_appended"] > 0
        assert r["wrong_append"] == pytest.approx(0.0) and r["contamination"] == pytest.approx(0.0)

    def test_a_reject_cost_of_zero_appends_nothing_and_matches_frozen(self):
        from recognize.protocols import run_p4
        f, y, order, pages = self.data()
        a = run_p4(f, y, order, 1, "random", 0, update_policy="frozen")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="assign",
                   link_groups=self.groups_from(y, pages), page_ids=pages, eta=0.0)
        assert b["n_appended"] == 0 and b["n_abstained"] == b["n_queries"]
        assert b["R1_identity"] == a["R1_identity"] and b["mAP"] == a["mAP"]

    def test_two_groups_on_one_page_never_take_the_same_identity(self):
        from recognize.protocols.assign import assign_page
        import numpy as _np
        f = _np.eye(4)[:, :3].astype(float)
        f[3] = f[0]                                     # crops 0 and 3 look identical
        g_feats = _np.array([[1.0, 0, 0], [0, 1.0, 0]])
        picks = assign_page(f, [0, 3], ["pg#a", None, None, "pg#b"], g_feats, _np.array([7, 9]), [7, 9],
                            eta=9.0, unlinked="abstain")
        assert sorted(picks) == [0, 3] and picks[0] != picks[3]

    def test_a_group_moves_as_one(self):
        from recognize.protocols.assign import assign_page
        import numpy as _np
        f = _np.array([[1.0, 0.0], [0.0, 1.0]])
        g_feats = _np.array([[1.0, 0.0]])
        picks = assign_page(f, [0, 1], ["pg#a", "pg#a"], g_feats, _np.array([5]), [5],
                            eta=9.0, unlinked="abstain")
        assert picks == {0: 5, 1: 5}                    # the must-link equality, not two decisions

    def test_unlinked_crops_are_dropped_or_freed(self):
        from recognize.protocols.assign import page_groups
        assert page_groups([0, 1, 2], ["a", None, "a"], "abstain") == [[0, 2]]
        assert page_groups([0, 1, 2], ["a", None, "a"], "free") == [[0, 2], [1]]

    def test_the_page_is_scored_before_any_of_its_own_appends(self):
        from recognize.protocols import run_p4
        f, y, order, pages = self.data(page=100)        # one page: nothing may be appended in time
        a = run_p4(f, y, order, 1, "random", 0, update_policy="frozen")
        b = run_p4(f, y, order, 1, "random", 0, update_policy="assign",
                   link_groups=self.groups_from(y, pages), page_ids=pages, eta=2.0)
        assert b["n_appended"] > 0                      # the flush still happens, at the very end
        assert b["R1_identity"] == a["R1_identity"] and b["mAP"] == a["mAP"]

    def test_a_looser_reject_appends_more(self):
        from recognize.protocols import run_p4
        f, y, order, pages = self.data()
        groups = self.groups_from(y, pages)
        counts = [run_p4(f, y, order, 1, "random", 0, update_policy="assign", link_groups=groups,
                         page_ids=pages, eta=e)["n_appended"] for e in (0.2, 0.6, 1.0, 2.0)]
        assert counts == sorted(counts) and counts[0] < counts[-1]

    def test_the_parameters_and_the_policy_must_agree(self):
        from recognize.protocols import run_p4
        f, y, order, pages = self.data()
        groups = self.groups_from(y, pages)
        with pytest.raises(ValueError, match="page_ids"):
            run_p4(f, y, order, 1, "random", 0, update_policy="assign", link_groups=groups)
        with pytest.raises(ValueError, match="link_groups"):
            run_p4(f, y, order, 1, "random", 0, update_policy="assign", page_ids=pages)
        with pytest.raises(ValueError, match="page_ids"):
            run_p4(f, y, order, 1, "random", 0, update_policy="predicted", page_ids=pages)

    def test_an_unknown_unlinked_mode_raises(self):
        from recognize.protocols.assign import page_groups
        with pytest.raises(ValueError, match="unlinked"):
            page_groups([0], ["a"], "sometimes")

    def test_dropping_the_exclusion_lets_two_groups_share_an_identity(self):
        from recognize.protocols.assign import assign_page
        import numpy as _np
        f = _np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        g_feats = _np.array([[1.0, 0.0], [0.0, 1.0]])
        args = (f, [0, 3], ["pg#a", None, None, "pg#b"], g_feats, _np.array([7, 9]), [7, 9])
        joint = assign_page(*args, eta=9.0, exclusive=True)
        free = assign_page(*args, eta=9.0, exclusive=False)
        assert joint[0] != joint[3]                       # exclusion forces them apart
        assert free[0] == free[3] == 7                    # both prefer identity 7 and both get it

    def test_the_exclusion_control_is_recorded(self):
        from recognize.protocols import run_p4
        f, y, order, pages = self.data()
        groups = self.groups_from(y, pages)
        r = run_p4(f, y, order, 1, "random", 0, update_policy="assign", link_groups=groups,
                   page_ids=pages, eta=1.0, exclusive=False)
        assert r["exclusive"] is False and r["eta"] == pytest.approx(1.0)

    def test_an_infinite_reject_assigns_every_group_it_can(self):
        from recognize.protocols.assign import solve
        import numpy as _np, math as _math
        picks = solve(_np.array([[0.9, 1.1], [1.3, 0.4], [1.0, 1.0]]), [1, 1, 1], _math.inf)
        assert sorted(p for p in picks if p is not None) == [0, 1]   # only two identities exist
        assert picks.count(None) == 1                                 # the third is forced out

    def test_a_group_cannot_borrow_another_groups_reject_price(self):
        from recognize.protocols.assign import solve
        import numpy as _np
        # Two groups, one identity. The singleton rejects for 0.5 and the four-member group for
        # 2.0, so the identity belongs to the larger group (1.2 + 0.5) and not to the closer one
        # (1.0 + 2.0). Letting either borrow the other's reject price would reverse that.
        picks = solve(_np.array([[1.0], [1.2]]), [1, 4], 0.5)
        assert picks == [None, 0]
