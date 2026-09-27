"""Evaluation manifest (scripts/slurm/make_eval_manifest.py) and array wiring."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "slurm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class TestManifest:
    def test_headline_group_covers_every_model(self):
        import make_eval_manifest as m
        from train import load_runs
        rows = m.build(["popcharacters"])
        assert len(rows) == 70 + 10                                   # every run plus 5 pretrained x {no memory, memory}
        tags = {r["tag"] for r in rows}
        assert {r["name"] for r in load_runs()} <= tags
        assert "pretrained__magiv2" in tags and "pretrained__magiv2__memory" in tags
        for r in rows:
            assert r["corpus"] == "popcharacters" and "--b-max-sweep" in r["extra"] and "--seeds 0 1 2 3 4" in r["extra"]
        assert all(r["kind"] in ("checkpoint", "pretrained") for r in rows)

    def test_other_groups_are_smaller_and_targeted(self):
        import make_eval_manifest as m
        m109 = m.build(["manga109"]); rev = m.build(["reverse"]); rob = m.build(["perturbations"])
        assert len(m109) == 20 and all(r["corpus"] == "manga109" and "--k 1 3 5" in r["extra"] for r in m109)
        assert len(rev) == 8 and all(r["corpus"] == "reverse" and r["extra"].startswith("--protocols p1") for r in rev)
        assert len(rob) == 4 * 11                                     # 2 backbones x {finetuned, memory} x 11 perturbations
        assert {k for r in rob for k in ("--box-noise", "--pixel-noise") if k in r["extra"]} == {"--box-noise", "--pixel-noise"}

    def test_round_trip(self, tmp_path):
        import make_eval_manifest as m
        rows = m.build(["popcharacters"])
        m.write(rows, tmp_path / "m.tsv")
        assert m.read(tmp_path / "m.tsv") == rows


class TestSbatch:
    def test_array_script_pins_the_environment(self):
        text = (REPO / "scripts" / "slurm" / "eval.sbatch").read_text()
        for needle in ("export PYTHONHASHSEED=0", "HF_HUB_OFFLINE=1", "#SBATCH --requeue",
                       "--reuse-cached", "EVAL_MANIFEST", "scripts/evaluate.py"):
            assert needle in text, needle

    def test_submit_script_excludes_pascal_and_stamps_the_manifest(self):
        text = (REPO / "scripts" / "slurm" / "submit_eval.sh").read_text()
        assert "pascal" in text and "--exclude=" in text and "eval_manifest_${STAMP}.tsv" in text


class TestAlternativeManifestRows:
    """Another run list is evaluated by the same machinery."""

    SEEDS = Path(__file__).resolve().parents[1] / "configs" / "runs_seeds.yaml"

    def test_it_covers_the_list_and_nothing_else(self):
        import make_eval_manifest as m
        rows = m.build(["popcharacters"], self.SEEDS)
        assert len(rows) == len(m.load_runs(self.SEEDS)) == 20
        assert all(r["kind"] == "checkpoint" for r in rows)          # no pretrained rows
        assert {r["tag"] for r in rows} == {r["name"] for r in m.load_runs(self.SEEDS)}

    def test_the_grid_is_the_same_one_the_campaign_uses(self):
        import make_eval_manifest as m
        assert {r["extra"] for r in m.build(["popcharacters"], self.SEEDS)} == {m.FULL_GRID}

    def test_without_it_the_campaign_is_unchanged(self):
        import make_eval_manifest as m
        tags = {r["tag"] for r in m.build(["popcharacters"])}
        assert "magiv2_lora_seed1" not in tags and "pretrained__magiv2" in tags
