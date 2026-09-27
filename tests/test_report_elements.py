"""Per-element, per-protocol report: the three kinds of difference it can form."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "report"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_report_tables import _result  # noqa: E402


class TestDelta:
    def test_two_three_seed_cells_are_paired(self):
        import elements
        d = elements.delta({0: 1.0, 1: 2.0, 2: 3.0}, {0: 2.0, 1: 3.0, 2: 4.0})
        assert d["kind"] == "paired" and d["delta"] == pytest.approx(1.0)
        assert d["sd"] == pytest.approx(0.0) and d["p"] is not None

    def test_the_pretrained_side_has_no_seed_so_each_seed_is_differenced_against_it(self):
        import elements
        d = elements.delta({0: 10.0}, {0: 11.0, 1: 12.0, 2: 13.0}, before_deterministic=True)
        assert d["kind"] == "fixed" and d["delta"] == pytest.approx(2.0) and d["n"] == 3
        assert d["p"] is not None                       # a one-sample test is legitimate here

    def test_a_single_seed_cell_gets_no_p_value(self):
        import elements
        d = elements.delta({0: 1.0, 1: 2.0, 2: 3.0}, {0: 5.0})
        assert d["kind"] == "one-seed" and d["p"] is None
        assert d["delta"] == pytest.approx(5.0 - 2.0)   # against the three-seed mean, not seed 0
        assert d["sd"] is None

    def test_an_empty_side_gives_nothing(self):
        import elements
        assert elements.delta({}, {0: 1.0}) is None and elements.delta({0: 1.0}, {}) is None


class TestFormatting:
    def test_each_kind_carries_its_own_mark(self):
        import elements
        assert elements.fmt({"delta": 1.0, "p": 0.5, "kind": "paired"}, 2) == "+1.00"
        assert elements.fmt({"delta": 1.0, "p": 0.5, "kind": "fixed"}, 2) == "+1.00‡"
        assert elements.fmt({"delta": 1.0, "p": None, "kind": "one-seed"}, 2) == "+1.00†"
        assert elements.fmt({"delta": -1.0, "p": 0.01, "kind": "paired"}, 2) == "**-1.00**"
        assert elements.fmt(None, 2) == "n/a"


class TestSummaryPolarity:
    def test_fewer_p3_clusters_counts_as_an_improvement(self):
        import elements
        rec = {}
        for label, _p, _kw, _s, _d, _h in elements.METRICS:
            for bb in elements.BACKBONES:
                for elem, _b, _a in elements.ELEMENTS:
                    sign = -1.0 if label == "P3 clusters" else 1.0
                    rec[(label, bb, elem)] = {"delta": sign * 2.0, "p": None, "kind": "paired",
                                              "n": 3, "sd": 0.0}
        text = elements.summary_table(rec)
        # every element helps on every backbone, including the column where lower is better
        assert "(5/5)" in text and "(0/5)" not in text
        assert "-2.0 (5/5)" in text


class TestEndToEnd:
    def test_it_writes_a_table_per_protocol(self, tmp_path, monkeypatch):
        import elements
        import tables
        series = ["Bakuman", "Dr Stone"]
        monkeypatch.setattr(elements, "load_split", lambda: {"test": series})
        root = tmp_path / "popcharacters"
        for tag, val in (("pretrained__magiv2", 0.40), ("magiv2_finetuned_seed0", 0.450),
                         ("magiv2_finetuned_seed1", 0.451), ("magiv2_finetuned_seed2", 0.449),
                         ("magiv2_memory_seed0", 0.500), ("magiv2_memory_seed1", 0.502),
                         ("magiv2_memory_seed2", 0.498)):
            d = root / tag; d.mkdir(parents=True)
            for s in series:
                (d / f"{tables.safe_name(s)}.json").write_text(json.dumps(_result({"0": val})))
        out = tmp_path / "e.md"
        assert elements.main(["--results", str(root), "--out", str(out)]) == 0
        text = out.read_text()
        for label, *_ in elements.METRICS:
            assert f"### {label}" in text
        assert "+5.00**‡" in text                        # 45.0 - 40.0, pretrained has no seed
        assert "+5.00**" in text.split("### P1 mAP")[1]  # memory - finetuned, paired
        assert "| magiv2 | " in text and "n/a" in text   # the other backbones have no results

    def test_a_missing_results_tree_is_refused(self, tmp_path, capsys):
        import elements
        assert elements.main(["--results", str(tmp_path / "nope"), "--out", str(tmp_path / "x.md")]) == 1
