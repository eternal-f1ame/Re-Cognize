"""Result-table builder (scripts/report/tables.py), on synthetic result JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "report"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _result(p1_by_seed, p2=0.3, p4=0.4, purity=0.7, clusters=12):
    return {"provenance": {"git_commit": "abc1234"},
            "series": {"name": "X", "n_crops": 10, "n_identities": 3},
            "p1": {s: {"mAP": v, "R1": v - 0.1} for s, v in p1_by_seed.items()},
            "p2": {st: {"1": {s: {"mAP": p2} for s in p1_by_seed}} for st in ("random", "temporal")},
            "p4": {st: {"1": {s: {"predicted": {"50": {"mAP": p4, "R1_identity": p4 + 0.05}}} for s in p1_by_seed}}
                   for st in ("random", "temporal")},
            "p3": {"fixed": {"purity": purity, "clusters": clusters}}}


@pytest.fixture
def tree(tmp_path):
    import tables
    series = ["Bakuman", "Dr Stone"]
    root = tmp_path / "popcharacters"
    for tag, base in (("transreid_finetuned_seed0", 0.40), ("pretrained__transreid", 0.30)):
        d = root / tag; d.mkdir(parents=True)
        for i, s in enumerate(series):
            (d / f"{tables.safe_name(s)}.json").write_text(json.dumps(
                _result({"0": base + 0.1 * i, "1": base + 0.1 * i + 0.02})))
    return tables, root, series


class TestCell:
    def test_mean_over_seeds_of_the_macro_over_series(self, tree):
        tables, root, series = tree
        c = tables.cell(tables.load_series(root, "transreid_finetuned_seed0", series), "p1")
        # seed 0 macro = (0.40 + 0.50)/2 = 0.45; seed 1 macro = 0.47 -> mean 0.46, sd 0.01414
        assert c["mean"] == pytest.approx(0.46) and c["sd"] == pytest.approx(0.01414, abs=1e-4)
        assert c["n_series"] == 2 and c["n_seeds"] == 2

    def test_missing_tag_gives_none(self, tree):
        tables, root, series = tree
        assert tables.cell(tables.load_series(root, "nothing_here", series), "p1") is None

    def test_p4_reads_policy_and_bmax(self, tree):
        tables, root, series = tree
        c = tables.cell(tables.load_series(root, "transreid_finetuned_seed0", series), "p4", metric="R1_identity")
        assert c["mean"] == pytest.approx(0.45)

    def test_p3_is_seedless(self, tree):
        tables, root, series = tree
        c = tables.cell(tables.load_series(root, "transreid_finetuned_seed0", series), "p3", metric="clusters")
        assert c["mean"] == pytest.approx(12) and c["n_seeds"] == 1


class TestRender:
    def test_table_marks_absent_cells(self, tree):
        tables, root, series = tree
        rows = tables.grid(root, series, "p1")
        text = tables.render(rows, "P1")
        # pretrained: seed macros (0.30+0.40)/2 and +0.02 -> 36.0; finetuned likewise -> 46.0
        assert "| transreid | 36.0 ± 1.4 |" in text and "46.0 ± 1.4" in text
        assert text.count("n/a") >= 4                       # configurations with no results yet
        assert "magiv3" in text                             # every backbone gets a row

    def test_cli_refuses_a_missing_tree(self, tmp_path, capsys):
        import tables
        assert tables.main(["--results", str(tmp_path / "nope"), "--out", str(tmp_path / "o.md")]) == 1
        assert "does not exist yet" in capsys.readouterr().out

    def test_cli_writes_every_table(self, tree, tmp_path):
        tables, root, series = tree
        out = tmp_path / "tables.md"
        assert tables.main(["--results", str(root), "--out", str(out)]) == 0
        text = out.read_text()
        for title in ("P1 closed set, mAP", "P1 closed set, Rank-1", "P2 Seq-R at k=1", "P2 Seq-T at k=1",
                      "P4 Seq-R at k=1", "P3 full stream, purity", "P3 full stream, predicted clusters"):
            assert title in text, title
        assert "commit(s) abc1234" in text

    def test_cli_reads_results_that_record_no_commit(self, tree, tmp_path):
        """The released result files carry no repository commit; the report is built without one."""
        tables, root, series = tree
        for f in root.glob("*/*.json"):
            d = json.loads(f.read_text())
            d["provenance"].pop("git_commit")
            f.write_text(json.dumps(d))
        out = tmp_path / "tables.md"
        assert tables.main(["--results", str(root), "--out", str(out)]) == 0
        assert "P1 closed set, mAP" in out.read_text() and "commit(s)" not in out.read_text()


@pytest.fixture
def seeded_tree(tmp_path):
    """Three training seeds of magiv2 memory, and a no-WM ablation exactly 2 points below each."""
    import tables
    series = ["Bakuman", "Dr Stone"]
    root = tmp_path / "popcharacters"
    base = {0: 0.50, 1: 0.56, 2: 0.53}                       # a deliberately wide training-seed spread
    for seed, v in base.items():
        for cfg, off in (("memory", 0.0), ("ablation_no_wm", -0.02)):
            d = root / f"magiv2_{cfg}_seed{seed}"; d.mkdir(parents=True)
            for i, s in enumerate(series):
                (d / f"{tables.safe_name(s)}.json").write_text(json.dumps(
                    _result({"0": v + off + 0.01 * i, "1": v + off + 0.01 * i}, p2=v + off)))
    return tables, root, series


class TestTrainingSeedTables:
    def test_by_train_seed_finds_every_seed_that_exists(self, seeded_tree):
        tables, root, series = seeded_tree
        got = tables.by_train_seed(root, series, "magiv2", "memory", "p1")
        assert sorted(got) == [0, 1, 2]
        assert got[0] == pytest.approx(0.505)                # mean over the two series

    def test_seed_spread_reports_n_and_sd(self, seeded_tree):
        tables, root, series = seeded_tree
        text = tables.render_seed_spread(root, series)
        assert "| magiv2 | Finetuned+Memory | 3 |" in text
        assert "s0=50.50, s1=56.50, s2=53.50" in text
        assert "Largest training-seed sd on P1 mAP: 3.00 points" in text

    def test_the_spread_table_covers_the_ablations_too(self, seeded_tree):
        # the floor is estimated from every configuration with more than one seed, not just two
        tables, root, series = seeded_tree
        text = tables.render_seed_spread(root, series)
        assert "Ablation: no working memory" in text
        assert "over 2 configurations" in text

    def test_it_states_what_effect_size_three_seeds_can_resolve(self, seeded_tree):
        tables, root, series = seeded_tree
        assert "is not resolvable" in tables.render_seed_spread(root, series)

    def test_a_configuration_with_one_seed_is_not_given_an_sd(self, tree):
        tables, root, series = tree
        assert "transreid" not in tables.render_seed_spread(root, series)

    def test_the_ablation_delta_is_paired_so_seed_spread_cancels(self, seeded_tree):
        tables, root, series = seeded_tree
        text = tables.render_ablations(root, series, backbones=("magiv2",))
        row = [l for l in text.splitlines() if "no working memory" in l][0]
        # per-seed spread is 3.00 points, the paired delta is exactly -2.00 with no spread at all
        assert "-2.00 ± 0.00" in row and "| 3 |" in row

    def test_an_ablation_with_no_results_is_left_out_rather_than_guessed(self, seeded_tree):
        tables, root, series = seeded_tree
        text = tables.render_ablations(root, series, backbones=("magiv2",))
        assert "no episodic memory" not in text and "(full memory)" in text


class TestPartialEvaluationIsVisible:
    """A tag that has been evaluated on half its series must never pass for a result."""

    @pytest.fixture
    def partial(self, tmp_path):
        import tables
        root = tmp_path / "popcharacters"
        for tag, n in (("magiv2_memory_seed0", 2), ("magiv2_memory_seed1", 2),
                       ("magiv2_memory_seed2", 2), ("magiv2_finetuned_seed0", 1)):
            d = root / tag; d.mkdir(parents=True)
            for i in range(n):
                (d / f"{tables.safe_name(['Bakuman', 'Dr Stone'][i])}.json").write_text(
                    json.dumps(_result({"0": 0.5 + 0.02 * i})))
        return tables, root, ["Bakuman", "Dr Stone"]

    def test_a_short_cell_is_labelled_in_the_grid(self, partial):
        tables, root, series = partial
        rows = tables.grid(root, series, "p1")
        text = tables.render(rows, "t", n_series=len(series))
        assert "(1/2 series)" in text                       # the finetuned tag has one of two
        assert "(2/2 series)" not in text                   # a complete cell is not decorated

    def test_a_short_seed_is_left_out_of_the_spread(self, partial):
        tables, root, series = partial
        assert sorted(tables.by_train_seed(root, series, "magiv2", "memory", "p1")) == [0, 1, 2]
        # ask for more series than exist and every seed drops out rather than being averaged short
        assert tables.by_train_seed(root, series + ["Ghost"], "magiv2", "memory", "p1") == {}

    def test_the_spread_table_omits_a_configuration_it_cannot_complete(self, partial):
        tables, root, series = partial
        assert "magiv2" not in tables.render_seed_spread(root, series + ["Ghost"])


class TestCorpusOption:
    """The same builder makes the Manga109 grid; only the series and the tag suffix change."""

    def test_manga109_tags_carry_the_suffix(self, tmp_path, monkeypatch):
        import tables
        series = ["ARMS", "MiraiSan"]
        root = tmp_path / "manga109"
        d = root / "magiv2_memory_seed0__manga109"; d.mkdir(parents=True)
        for s in series:
            (d / f"{tables.safe_name(s)}.json").write_text(json.dumps(_result({"0": 0.66})))
        monkeypatch.setattr(tables, "load_split", lambda: {"test": ["unused"]})
        out = tmp_path / "t.md"
        rc = tables.main(["--corpus", "manga109", "--results", str(root), "--out", str(out)])
        assert rc == 0
        text = out.read_text()
        assert "Manga109, zero-shot" in text and "66.0" in text
        assert "ARMS" in text

    def test_the_popcharacters_default_is_untouched(self, tree, tmp_path):
        tables, root, series = tree
        out = tmp_path / "p.md"
        assert tables.main(["--results", str(root), "--out", str(out)]) == 0
        assert "POPCharacters" in out.read_text()


class TestMissingMetric:
    """A result file without the requested metric gives None, not a division by zero."""

    def test_asking_for_a_metric_no_file_has(self, tree):
        tables, root, series = tree
        # the synthetic P3 block has purity and clusters, never ari
        assert tables.cell(tables.load_series(root, "transreid_finetuned_seed0", series),
                           "p3", "ari") is None

    def test_the_metrics_that_are_there_still_work(self, tree):
        tables, root, series = tree
        c = tables.cell(tables.load_series(root, "transreid_finetuned_seed0", series), "p3", "purity")
        assert c is not None and c["mean"] == pytest.approx(0.7)


class TestCorpusSuffixDoesNotLeak:
    """The Manga109 tag suffix travels as an argument; it must not rebind anything module-level."""

    def test_tag_for_is_the_same_function_after_a_manga109_build(self, tmp_path, monkeypatch):
        import tables
        before = tables.tag_for("magiv2", "memory")
        root = tmp_path / "manga109"
        d = root / "magiv2_memory_seed0__manga109"; d.mkdir(parents=True)
        (d / f"{tables.safe_name('ARMS')}.json").write_text(json.dumps(_result({"0": 0.5})))
        monkeypatch.setattr(tables, "load_split", lambda: {"test": ["ARMS"]})
        assert tables.main(["--corpus", "manga109", "--results", str(root),
                            "--out", str(tmp_path / "m.md")]) == 0
        assert tables.tag_for("magiv2", "memory") == before == "magiv2_memory_seed0"

    def test_the_suffix_reaches_every_helper(self, tmp_path):
        import tables
        assert tables.tag_for("magiv2", "memory", 1, "__manga109") == "magiv2_memory_seed1__manga109"
        assert tables.tag_for("magiv2", "pretrained", suffix="__manga109") == "pretrained__magiv2__manga109"


class TestCorpusPathsAndEmptyGuard:
    """`--corpus` alone must read and write that corpus's own files, and never write an empty grid."""

    def test_corpus_selects_both_the_results_tree_and_the_output_file(self):
        from report.tables import default_paths
        res, out = default_paths("popcharacters")
        assert res.name == "popcharacters" and out.name == "tables.md"
        res, out = default_paths("manga109")
        assert res.name == "manga109" and out.name == "tables_manga109.md"

    def test_a_grid_with_no_measurements_is_refused(self, tmp_path, capsys):
        from report import tables
        empty = tmp_path / "results"
        empty.mkdir()
        rc = tables.main(["--corpus", "manga109",
                          "--results", str(empty),
                          "--out", str(tmp_path / "x.md")])
        assert rc == 1
        assert not (tmp_path / "x.md").exists()
        assert "every cell is n/a" in capsys.readouterr().out
