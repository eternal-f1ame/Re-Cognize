"""Crop-robustness report (scripts/report/robustness.py), on synthetic result JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "report"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_report_tables import _result  # noqa: E402  (same synthetic shape)


@pytest.fixture
def tree(tmp_path):
    import robustness
    import tables
    series = ["Bakuman", "Dr Stone"]
    root = tmp_path / "popcharacters"
    # clean: memory 1 point above finetuned; occluded: memory 3 points above
    for tag, p1 in (("transreid_finetuned_seed0", 0.40), ("transreid_memory_seed0", 0.41),
                    ("transreid_finetuned_seed0__occ30", 0.30), ("transreid_memory_seed0__occ30", 0.33),
                    ("magiv2_finetuned_seed0", 0.50), ("magiv2_memory_seed0", 0.51)):
        d = root / tag; d.mkdir(parents=True)
        for s in series:
            (d / f"{tables.safe_name(s)}.json").write_text(json.dumps(_result({"0": p1}, p2=p1)))
    return robustness, root, series


class TestRobustnessTable:
    def test_the_clean_row_is_absolute_and_the_rest_are_deltas(self, tree):
        rob, root, series = tree
        text = rob.table(root, series, "p1", "t")
        assert "| clean | 40.00 | 41.00 | 50.00 | 51.00 | +1.00 | +1.00 |" in text
        assert "| occ30 | -10.00 | -8.00 | n/a | n/a | +3.00 | n/a |" in text

    def test_a_perturbation_nobody_ran_is_left_out(self, tree):
        rob, root, series = tree
        assert "blur4" not in rob.table(root, series, "p1", "t")

    def test_the_summary_says_whether_the_gap_grows(self, tree):
        rob, root, series = tree
        text = rob.summary(root, series)
        assert "transreid" in text and "earns more as the crop degrades" in text
        # magiv2 has no perturbed tags at all, so it must not be given a verdict
        assert "**magiv2**" not in text

    def test_a_short_tag_is_not_averaged_into_a_delta(self, tree, tmp_path):
        rob, root, series = tree
        assert rob.value(root, "transreid_memory_seed0", series + ["Ghost"], "p1") is None
