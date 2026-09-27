"""Every reportable checkpoint must name code that existed. Enforced, not just intended.

Submission checks the tree (`scripts/train.py:assert_clean_tree`), but an array task starts minutes
or hours later and reads whatever is on disk then, so a module edited while the array waits in the
queue leaves checkpoints that record `git_dirty`. This test is the backstop: a dirty checkpoint
fails the suite, which blocks the commit that would report it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

MANIFESTS = [REPO / "configs" / "runs.yaml", REPO / "configs" / "runs_seeds.yaml"]


def provenances():
    import train
    out = []
    for m in MANIFESTS:
        for run in train.load_runs(m):
            p = train.output_dir(run) / "provenance.json"
            if p.exists():
                out.append((run["name"], p, json.loads(p.read_text())))
    return out


class TestCheckpointProvenance:
    def test_no_reportable_checkpoint_was_trained_from_a_modified_tree(self):
        rows = provenances()
        if not rows:
            pytest.skip("no checkpoints in this checkout")
        dirty = [(n, d.get("git_dirty_files")) for n, _, d in rows if d.get("git_dirty")]
        assert not dirty, f"{len(dirty)} checkpoint(s) trained from a modified tree: {dirty[:5]}"

    def test_every_checkpoint_names_a_commit(self):
        rows = provenances()
        if not rows:
            pytest.skip("no checkpoints in this checkout")
        missing = [n for n, _, d in rows if not d.get("git_commit")]
        assert not missing, missing[:5]

    def test_the_hash_seed_was_pinned(self):
        rows = provenances()
        if not rows:
            pytest.skip("no checkpoints in this checkout")
        unpinned = [n for n, _, d in rows if str(d.get("pythonhashseed")) != "0"]
        assert not unpinned, unpinned[:5]
