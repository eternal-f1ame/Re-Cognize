"""The training run manifest and the launcher that turns a row into a command."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(scope="module")
def launcher():
    import train
    return train


def _flag(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


class TestManifest:
    def test_campaign_shape(self, launcher):
        runs = launcher.load_runs()
        groups = {}
        for r in runs:
            groups.setdefault(r["group"], []).append(r["name"])
        assert len(runs) == 70 and len({r["name"] for r in runs}) == 70
        assert len(groups["grid"]) == 20 and len(groups["ablation"]) == 30 and len(groups["seed"]) == 20
        # the ablations that answer a question run on the backbones whose memory has an effect
        for bb in ("magiv2", "magiv3"):
            cells = {r["config"] for r in runs if r["backbone"] == bb and r["group"] == "ablation"}
            assert cells == {"ablation_no_wm", "ablation_no_em", "ablation_no_id_drop", "ablation_no_mem_loss"}
            for cfg in cells | {"memory"}:
                seeds = sorted(r["seed"] for r in runs if r["backbone"] == bb and r["config"] == cfg)
                assert seeds == [0, 1, 2], (bb, cfg, seeds)
        assert {r["backbone"] for r in runs} == set(launcher.BACKBONE_REGISTRY)
        # every backbone gets the four grid cells at seed 0
        for bb in launcher.BACKBONE_REGISTRY:
            cells = {r["config"] for r in runs if r["backbone"] == bb and r["group"] == "grid"}
            assert cells == {"finetuned", "memory", "lora", "memory_lora"}
        # the headline cells (memory against finetuned) have three seeds on every backbone,
        # because the Rank-1 half of that comparison is only 1.2 to 2.4 times its floor
        for bb in launcher.BACKBONE_REGISTRY:
            for cfg in ("finetuned", "memory"):
                seeds = sorted(r["seed"] for r in runs if r["backbone"] == bb and r["config"] == cfg)
                assert seeds == [0, 1, 2], (bb, cfg, seeds)
        assert sorted(r["priority"] for r in runs) == list(range(1, len(runs) + 1))

    def test_ablations_carry_their_deviation(self, launcher):
        by = {r["name"]: r for r in launcher.load_runs()}
        assert by["transreid_ablation_no_wm_seed0"]["no_working_memory"] is True
        assert by["transreid_ablation_no_em_seed0"]["no_episodic_memory"] is True
        assert by["transreid_ablation_no_id_drop_seed0"]["episodic_id_drop_rate"] == 0.0
        assert by["transreid_ablation_no_mem_loss_seed0"]["memory_weight"] == 0.0
        assert by["transreid_ablation_lora_r4_seed0"]["lora_rank"] == 4 and by["transreid_ablation_lora_r4_seed0"]["lora"] is True
        assert by["transreid_ablation_lora_r16_seed0"]["lora_rank"] == 16
        assert all(r["memory"] for n, r in by.items() if "ablation" in n)


class TestCommand:
    def test_memory_run_uses_recipe_and_registry(self, launcher):
        from recognize.recipe import RECIPE
        cmd = launcher.build_command(launcher.get_run("transreid_memory_seed0"))
        spec = launcher.BACKBONE_REGISTRY["transreid"]
        assert _flag(cmd, "--epochs") == str(RECIPE.epochs) and _flag(cmd, "--lr") == str(RECIPE.lr)
        assert _flag(cmd, "--k") == str(spec.k) == "4" and _flag(cmd, "--k-support") == str(RECIPE.k_support) == "2"
        assert _flag(cmd, "--p") == str(spec.p) and _flag(cmd, "--feat-dim") == str(spec.native_dim)
        assert _flag(cmd, "--normalize") == spec.normalize and _flag(cmd, "--memory-lr-scale") == "1.0"
        assert _flag(cmd, "--ce-weight") == "0" and _flag(cmd, "--memory-weight") == str(RECIPE.w_mem)
        assert "--no-memory" not in cmd and "--amp" in cmd and _flag(cmd, "--seed") == "0"
        assert _flag(cmd, "--split-config").endswith("data_split.yaml") and "--resume" in cmd
        assert _flag(cmd, "--output-dir").endswith("checkpoints/transreid/memory/seed0")

    def test_baseline_disables_memory_and_its_loss(self, launcher):
        from recognize.recipe import RECIPE
        cmd = launcher.build_command(launcher.get_run("magiv3_finetuned_seed0"))
        assert "--no-memory" in cmd and _flag(cmd, "--memory-weight") == "0" and _flag(cmd, "--ce-weight") == str(RECIPE.w_ce)
        assert _flag(cmd, "--p") == "4" and _flag(cmd, "--feat-dim") == "1024"     # magiv3 registry values

    def test_lora_and_reid5o_specifics(self, launcher):
        from recognize.recipe import RECIPE
        cmd = launcher.build_command(launcher.get_run("magiv2_memory_lora_seed0"))
        assert "--use-lora" in cmd and _flag(cmd, "--lora-rank") == str(RECIPE.lora_rank) and _flag(cmd, "--lora-lr") == str(RECIPE.lora_lr)
        cmd5 = launcher.build_command(launcher.get_run("reid5o_memory_seed0"))
        assert _flag(cmd5, "--normalize") == "clip" and _flag(cmd5, "--reid5o-config").endswith("configs.yaml")

    def test_ablation_overrides_the_recipe_value(self, launcher):
        cmd = launcher.build_command(launcher.get_run("transreid_ablation_lora_r4_seed0"))
        assert _flag(cmd, "--lora-rank") == "4" and cmd.count("--lora-rank") == 1
        cmd = launcher.build_command(launcher.get_run("transreid_ablation_no_mem_loss_seed0"))
        assert _flag(cmd, "--memory-weight") == "0.0" and cmd.count("--memory-weight") == 1
        cmd = launcher.build_command(launcher.get_run("transreid_ablation_no_wm_seed0"))
        assert "--no-working-memory" in cmd
        cmd = launcher.build_command(launcher.get_run("transreid_ablation_no_id_drop_seed0"))
        assert _flag(cmd, "--episodic-id-drop-rate") == "0.0"

    def test_smoke_and_timing_shorten_the_run(self, launcher):
        smoke = launcher.build_command(launcher.get_run("transreid_memory_seed0"), mode="smoke")
        timing = launcher.build_command(launcher.get_run("transreid_memory_seed0"), mode="timing")
        assert _flag(smoke, "--epochs") == "2" and _flag(smoke, "--dev-eval-every") == "1"
        assert _flag(timing, "--epochs") == "1"
        assert "checkpoints/smoke/" in _flag(smoke, "--output-dir") and "checkpoints/timing/" in _flag(timing, "--output-dir")

    def test_every_run_builds(self, launcher):
        for r in launcher.load_runs():
            cmd = launcher.build_command(r)
            assert _flag(cmd, "--seed") == str(r["seed"]) and r["name"] == _flag(cmd, "--name")


class TestSbatch:
    def test_script_pins_the_environment(self):
        text = (REPO / "scripts" / "slurm" / "train.sbatch").read_text()
        for needle in ("export PYTHONHASHSEED=0", "HF_HUB_OFFLINE=1", "#SBATCH --requeue", "RUN_LIST", "scripts/train.py \"${MANIFEST_FLAG[@]}\" run"):
            assert needle in text, needle

    def test_short_partition_asks_for_turing(self, launcher):
        runs = launcher.load_runs()[:2]
        cmd = launcher.sbatch_command(runs, partition="short", mode="smoke", exclude="c1-3")
        assert "--gres=gpu:turing:1" in cmd and "--array=0,1" in cmd and "--exclude=c1-3" in cmd
        assert "--partition=short" in cmd and cmd[-1].endswith("train.sbatch")

    def test_dry_run_writes_nothing(self, launcher, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(launcher, "CHECKPOINTS", tmp_path / "ckpt")
        monkeypatch.setattr(launcher, "assert_clean_tree", lambda: "abc1234")
        assert launcher.main(["sbatch", "--runs", "transreid_memory_seed0", "--dry-run"]) == 0
        assert not (tmp_path / "ckpt").exists() and "sbatch" in capsys.readouterr().out

    def test_each_submission_gets_its_own_run_list(self, launcher, tmp_path, monkeypatch):
        """Array tasks read the list at execution time; a second submit must not overwrite the first."""
        monkeypatch.setattr(launcher, "CHECKPOINTS", tmp_path / "ckpt")
        monkeypatch.setattr(launcher, "assert_clean_tree", lambda: "abc1234")
        monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
        seen = []
        real = launcher.write_run_list
        monkeypatch.setattr(launcher, "write_run_list", lambda runs, path: (seen.append(path), real(runs, path))[1])
        launcher.main(["sbatch", "--smoke", "--runs", "transreid_memory_seed0"])
        launcher.main(["sbatch", "--smoke", "--runs", "magiv2_memory_seed0"])
        assert len(seen) == 2 and seen[0] != seen[1]
        assert seen[0].read_text().strip() == "transreid_memory_seed0"
        assert seen[1].read_text().strip() == "magiv2_memory_seed0"


class TestCleanTreeGuard:
    """A checkpoint's provenance must name code that exists, so a modified tree is refused."""

    def test_dirty_tree_is_refused(self, launcher, monkeypatch, tmp_path):
        monkeypatch.setattr(launcher, "CHECKPOINTS", tmp_path / "ckpt")
        monkeypatch.setattr(launcher.subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 0, "stdout": " M src/x.py\n"})())
        with pytest.raises(SystemExit, match="modified working tree"):
            launcher.main(["sbatch", "--runs", "transreid_memory_seed0"])

    def test_allow_dirty_overrides(self, launcher, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(launcher, "CHECKPOINTS", tmp_path / "ckpt")
        monkeypatch.setattr(launcher.subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "abc1234\n"})())
        assert launcher.main(["sbatch", "--runs", "transreid_memory_seed0", "--allow-dirty", "--dry-run"]) == 0
        assert "at commit abc1234" in capsys.readouterr().out


class TestReportedCheckpoint:
    """The reported checkpoint is the last epoch; dev selection is kept only for diagnostics."""

    def test_prefers_final_then_last_epoch_then_best(self, launcher, tmp_path, monkeypatch):
        from recognize.recipe import RECIPE
        run = launcher.get_run("transreid_memory_seed0")
        d = tmp_path / "transreid" / "memory" / "seed0"; d.mkdir(parents=True)
        monkeypatch.setattr(launcher, "output_dir", lambda r, mode="": d)
        (d / "best.pth").touch()
        assert launcher.reported_checkpoint(run).name == "best.pth"
        (d / f"epoch_{RECIPE.epochs:04d}.pth").touch()
        assert launcher.reported_checkpoint(run).name == f"epoch_{RECIPE.epochs:04d}.pth"
        (d / "final.pth").touch()
        assert launcher.reported_checkpoint(run).name == "final.pth"

    def test_the_eval_manifest_uses_it(self):
        import sys
        if str(REPO / "scripts" / "slurm") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts" / "slurm"))
        import make_eval_manifest as m
        rows = [r for r in m.build(["popcharacters"]) if r["kind"] == "checkpoint"]
        assert rows and all(r["arg"].endswith(("final.pth", "epoch_0200.pth", "best.pth")) for r in rows)
        # a run that has actually been trained is never reported from its dev-selected checkpoint
        trained = [r for r in rows if Path(r["arg"]).parent.joinpath("history.json").exists()]
        if not trained:
            pytest.skip("no trained checkpoints under checkpoints/")
        assert all(not r["arg"].endswith("best.pth") for r in trained)


class TestNodeSelection:
    """Placement rules: no Pascal (no sm_61 kernels) and no GPU smaller than the run needs."""

    SINFO = ("c1-6|gpu:pascal:4,gpumem:no_consume:11G|pascal,gmem11\n"
             "c2-0|gpu:turing:3|gmem12,gmemT48\n"
             "c3-2|gpu:turing:4,gpumem:no_consume:11G|turing,gmem11,infiniband\n"
             "c3-5|gpu:turing:3,gpumem:no_consume:16G|turing,gmem11,gmem12,gmem16,infiniband\n"
             "c9-9|gpu:turing:1|turing\n")

    @pytest.fixture()
    def sinfo(self, launcher, monkeypatch):
        rows = [tuple(l.split("|")) for l in self.SINFO.strip().split("\n")]
        monkeypatch.setattr(launcher, "_sinfo_nodes", lambda partition: rows)
        return launcher

    def test_gmem_features_are_read_as_a_maximum_not_a_ladder(self, launcher):
        assert launcher.node_gpu_mem_gb("turing,gmem11,gmem12,gmem16,infiniband") == 16
        assert launcher.node_gpu_mem_gb("gmem12,gmemT48") == 12
        assert launcher.node_gpu_mem_gb("turing,infiniband") is None

    def test_pascal_is_always_excluded(self, sinfo):
        assert "c1-6" in sinfo.unusable_nodes(11, "short").split(",")

    def test_small_gpus_are_excluded_for_a_big_run(self, sinfo):
        assert sinfo.unusable_nodes(16, "short").split(",") == ["c1-6", "c2-0", "c3-2", "c9-9"]

    def test_an_11gb_run_keeps_every_turing_node_that_advertises_its_size(self, sinfo):
        kept = {"c2-0", "c3-2", "c3-5"} - set(sinfo.unusable_nodes(11, "short").split(","))
        assert kept == {"c2-0", "c3-2", "c3-5"}

    def test_a_node_that_advertises_nothing_is_not_guessed_at(self, sinfo):
        assert "c9-9" in sinfo.unusable_nodes(11, "short").split(",")

    def test_every_campaign_run_states_a_size_some_node_can_serve(self, launcher):
        # 16 GB is the largest GPU on `short`; anything above it would exclude every node.
        assert all(r["gpu_mem_gb"] <= 16 for r in launcher.load_runs())


class TestEvalManifestFilter:
    """A half-trained campaign must be submittable without pointing tasks at missing checkpoints."""

    @pytest.fixture()
    def m(self):
        import sys
        if str(REPO / "scripts" / "slurm") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts" / "slurm"))
        import make_eval_manifest
        return make_eval_manifest

    def rows(self, tmp_path, names):
        out = []
        for name, files in names.items():
            d = tmp_path / name; d.mkdir()
            for f in files: (d / f).touch()
            out.append({"kind": "checkpoint", "tag": name, "arg": str(d / files[-1])})
        return out

    def test_a_finished_run_is_kept(self, m, tmp_path):
        rows = self.rows(tmp_path, {"done": ["best.pth", "epoch_0200.pth"]})
        keep, drop = m.drop_untrained(rows)
        assert [r["tag"] for r in keep] == ["done"] and not drop

    def test_a_run_still_training_is_dropped_even_though_best_pth_exists(self, m, tmp_path):
        # best.pth appears at the first dev evaluation, epoch 10 of 200
        rows = self.rows(tmp_path, {"midway": ["best.pth"]})
        keep, drop = m.drop_untrained(rows)
        assert not keep and [r["tag"] for r in drop] == ["midway"]

    def test_a_missing_checkpoint_is_dropped(self, m, tmp_path):
        keep, drop = m.drop_untrained([{"kind": "checkpoint", "tag": "x", "arg": str(tmp_path / "no.pth")}])
        assert not keep and len(drop) == 1

    def test_pretrained_rows_have_no_checkpoint_and_always_stay(self, m):
        keep, drop = m.drop_untrained([{"kind": "pretrained", "tag": "pretrained__magiv2", "arg": "magiv2"}])
        assert len(keep) == 1 and not drop


class TestAblationsActuallyAblate:
    """An ablation whose command matches its parent's measures nothing at all."""

    PARENT = {"ablation": "memory"}

    def parent_of(self, launcher, run):
        base = run["name"].replace(f"_{run['config']}_", "_memory_")
        return launcher.get_run(base)

    def test_every_ablation_differs_from_its_parent_in_the_command(self, launcher):
        seen = 0
        for run in launcher.load_runs():
            if not run["config"].startswith("ablation"):
                continue
            mine = launcher.build_command(run)
            theirs = launcher.build_command(self.parent_of(launcher, run))
            # ignore the parts that differ only because the run has another name
            strip = lambda c: [a for a in c if run["name"] not in a and "memory/seed" not in a
                               and f"{run['config']}/seed" not in a]
            assert strip(mine) != strip(theirs), f"{run['name']} trains exactly like its parent"
            seen += 1
        assert seen == 30                                   # 6 TransReID + 24 MagiV2/MagiV3

    def test_the_deviation_keys_all_map_to_a_flag(self, launcher):
        keys = {k for r in launcher.load_runs() for k in r
                if k not in ("name", "backbone", "config", "memory", "lora", "seed",
                             "priority", "gpu_mem_gb", "group", "note")}
        assert keys and keys <= set(launcher.DEVIATIONS)

    def test_a_zero_valued_deviation_is_passed_not_dropped(self, launcher):
        # memory_weight: 0.0 is falsy; a truthiness test in the builder would silently drop it
        cmd = launcher.build_command(launcher.get_run("magiv2_ablation_no_mem_loss_seed0"))
        assert cmd[cmd.index("--memory-weight") + 1] == "0.0"
        cmd = launcher.build_command(launcher.get_run("magiv2_ablation_no_id_drop_seed0"))
        assert cmd[cmd.index("--episodic-id-drop-rate") + 1] == "0.0"


class TestEvalManifestSkipsFinishedRows:
    """A finished tag should not cost a queue slot just to start and exit."""

    @pytest.fixture()
    def m(self):
        import sys
        if str(REPO / "scripts" / "slurm") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts" / "slurm"))
        import make_eval_manifest
        return make_eval_manifest

    def tag(self, root, name, n_series):
        d = root / "popcharacters" / name
        d.mkdir(parents=True)
        for i in range(n_series):
            (d / f"s{i}.json").write_text("{}")
        return {"kind": "checkpoint", "tag": name, "arg": "x.pth", "corpus": "popcharacters"}

    def test_a_tag_with_every_series_is_dropped(self, m, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "split_for", lambda corpus, split: ["a", "b", "c"])
        rows = [self.tag(tmp_path, "done", 3)]
        keep, done = m.drop_done(rows, tmp_path)
        assert not keep and [r["tag"] for r in done] == ["done"]

    def test_a_partial_tag_keeps_its_row(self, m, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "split_for", lambda corpus, split: ["a", "b", "c"])
        rows = [self.tag(tmp_path, "partial", 2)]
        keep, done = m.drop_done(rows, tmp_path)
        assert [r["tag"] for r in keep] == ["partial"] and not done

    def test_a_tag_with_no_results_keeps_its_row(self, m, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "split_for", lambda corpus, split: ["a", "b", "c"])
        rows = [{"kind": "checkpoint", "tag": "fresh", "arg": "x.pth", "corpus": "popcharacters"}]
        keep, done = m.drop_done(rows, tmp_path)
        assert [r["tag"] for r in keep] == ["fresh"]

    def test_a_corpus_with_no_test_series_is_never_called_finished(self, m, tmp_path, monkeypatch):
        # expected == 0 would otherwise make every row look complete
        monkeypatch.setattr(m, "split_for", lambda corpus, split: [])
        rows = [{"kind": "checkpoint", "tag": "x", "arg": "x.pth", "corpus": "popcharacters"}]
        keep, done = m.drop_done(rows, tmp_path)
        assert keep and not done


class TestAlternativeManifest:
    """A second run list (here the extra training runs of the LoRA cells) leaves the campaign's shape alone."""

    SEEDS = REPO / "configs" / "runs_seeds.yaml"

    def test_the_campaign_is_untouched_by_it(self, launcher):
        assert len(launcher.load_runs()) == 70
        assert all(r["group"] != "extra_seeds" for r in launcher.load_runs())

    def test_the_list_loads_and_names_are_unique(self, launcher):
        runs = launcher.load_runs(self.SEEDS)
        assert len(runs) == len({r["name"] for r in runs}) == 20
        assert all(r["group"] == "extra_seeds" and r["lora"] and r["seed"] in (1, 2) for r in runs)

    def test_its_checkpoints_do_not_collide_with_the_campaign(self, launcher):
        campaign = {launcher.output_dir(r) for r in launcher.load_runs()}
        extra = {launcher.output_dir(r) for r in launcher.load_runs(self.SEEDS)}
        assert not (campaign & extra)

    def test_an_unknown_group_is_refused_rather_than_selecting_nothing(self, launcher):
        with pytest.raises(SystemExit, match="unknown group"):
            launcher.main(["sbatch", "--groups", "nope", "--dry-run"])


class TestManifestReachesTheArrayTask:
    """An array built from another manifest must resolve its run names there, not in the campaign."""

    SBATCH = REPO / "scripts" / "slurm" / "train.sbatch"

    def test_the_sbatch_forwards_the_manifest(self):
        text = self.SBATCH.read_text()
        assert "RUN_MANIFEST" in text and '--manifest' in text
        assert 'scripts/train.py "${MANIFEST_FLAG[@]}" run' in text

    def test_the_launcher_exports_it(self):
        text = (REPO / "scripts" / "train.py").read_text()
        assert "RUN_MANIFEST=str(args.manifest)" in text

    def test_a_name_from_another_manifest_does_not_resolve_in_the_campaign(self, launcher):
        with pytest.raises(KeyError, match="unknown run"):
            launcher.get_run("magiv2_lora_seed1")
        assert launcher.get_run("magiv2_lora_seed1", launcher.load_runs(
            REPO / "configs" / "runs_seeds.yaml"))["config"] == "lora"
