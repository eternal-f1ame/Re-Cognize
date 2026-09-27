#!/usr/bin/env python3
"""Training launcher: one manifest row -> one training command.

    python scripts/train.py list                                  # the campaign
    python scripts/train.py command transreid_memory_seed0        # print the command
    python scripts/train.py run transreid_memory_seed0 [-- extra] # run it here
    python scripts/train.py sbatch [--runs A B] [--partition normal] [--smoke|--timing] [--dry-run]

The recipe lives in src/recognize/recipe.py, the backbone facts in
src/recognize/backbones.py and the campaign in configs/runs.yaml; this file only
turns a row into arguments. Checkpoints go to checkpoints/<backbone>/<config>/seed<s>/.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402

from recognize.backbones import BACKBONE_REGISTRY  # noqa: E402
from recognize.recipe import RECIPE  # noqa: E402

MANIFEST = PROJECT_ROOT / "configs" / "runs.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "memory_block" / "training" / "train.py"
SPLIT = PROJECT_ROOT / "configs" / "training" / "data_split.yaml"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "Datasets" / "popcharacters"
SBATCH = PROJECT_ROOT / "scripts" / "slurm" / "train.sbatch"
# Per-run deviations a manifest row may carry, mapped to their CLI flags.
DEVIATIONS = {
    "no_working_memory": "--no-working-memory",
    "no_episodic_memory": "--no-episodic-memory",
    "episodic_id_drop_rate": "--episodic-id-drop-rate",
    "memory_weight": "--memory-weight",
    "residual_max_ratio": "--residual-max-ratio",
    "lora_rank": "--lora-rank",
}


def load_runs(path: Path = MANIFEST) -> List[Dict]:
    runs = yaml.safe_load(Path(path).read_text())["runs"]
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate run names in the manifest")
    return runs


def get_run(name: str, runs: Optional[Sequence[Dict]] = None) -> Dict:
    runs = runs if runs is not None else load_runs()
    for r in runs:
        if r["name"] == name:
            return r
    raise KeyError(f"unknown run {name!r}; `python scripts/train.py list` shows all {len(runs)}")


def reported_checkpoint(run: Dict, mode: str = "") -> Path:
    """The checkpoint a run reports: its last epoch.

    Selecting on the dev set picks a checkpoint 0.10 to 0.39 mAP worse on test than the last epoch
    for every memory run, and for some runs picks epoch 10 of 200, so the reported checkpoint is the
    final one. Without final.pth the last epoch file is used, and best.pth only for a run that has
    not finished.
    """
    d = output_dir(run, mode)
    final = d / "final.pth"
    if final.exists():
        return final
    last = d / f"epoch_{RECIPE.epochs:04d}.pth"
    return last if last.exists() else d / "best.pth"


def output_dir(run: Dict, mode: str = "") -> Path:
    base = CHECKPOINTS / mode if mode else CHECKPOINTS
    return base / run["backbone"] / run["config"] / f"seed{run['seed']}"


def build_command(run: Dict, *, mode: str = "", extra: Sequence[str] = (), python: str = sys.executable) -> List[str]:
    """The full training command for one manifest row (recipe + registry + deviations)."""
    spec = BACKBONE_REGISTRY[run["backbone"]]
    out = output_dir(run, mode)
    epochs = {"smoke": 2, "timing": 1}.get(mode, RECIPE.epochs)
    dev_every = 1 if mode else RECIPE.dev_eval_every
    cmd = [
        python, str(TRAIN_SCRIPT),
        "--data-dir", str(DATA_DIR),
        "--split-config", str(SPLIT),
        "--backbone", run["backbone"],
        "--feat-dim", str(spec.native_dim),
        "--height", str(spec.height), "--width", str(spec.width),
        "--normalize", spec.normalize,
        "--epochs", str(epochs),
        "--lr", str(RECIPE.lr), "--weight-decay", str(RECIPE.weight_decay),
        "--warmup-epochs", str(RECIPE.warmup_epochs),
        "--pk-sampling", "--p", str(spec.p), "--k", str(spec.k), "--k-support", str(RECIPE.k_support),
        "--triplet-margin", str(RECIPE.triplet_margin), "--proto-temperature", str(RECIPE.proto_temperature),
        "--triplet-weight", str(RECIPE.w_trip), "--prototype-weight", str(RECIPE.w_proto),
        "--memory-lr-scale", str(RECIPE.memory_lr_scale),
        "--working-capacity", str(RECIPE.working_capacity), "--slots-per-char", str(RECIPE.slots_per_char),
        "--seed", str(run["seed"]),
        "--save-freq", str(RECIPE.save_every), "--dev-eval-every", str(dev_every),
        "--name", run["name"], "--output-dir", str(out),
        "--resume",
    ]
    cmd += ["--amp"] if RECIPE.amp else ["--no-amp"]
    if run["memory"]:
        # The classifier is discarded at inference; its gradients only add noise to the memory path.
        cmd += ["--ce-weight", "0", "--memory-weight", str(RECIPE.w_mem)]
    else:
        # No memory block: no memory-consistency term, and CE stays on.
        cmd += ["--no-memory", "--ce-weight", str(RECIPE.w_ce), "--memory-weight", "0"]
    if run["lora"]:
        cmd += ["--use-lora", "--lora-rank", str(RECIPE.lora_rank), "--lora-alpha", str(RECIPE.lora_alpha),
                "--lora-layers", str(RECIPE.lora_layers), "--lora-lr", str(RECIPE.lora_lr)]
    if spec.reid5o_config:
        cmd += ["--reid5o-config", str(PROJECT_ROOT / spec.reid5o_config)]
    for key, flag in DEVIATIONS.items():                       # ablations override recipe values
        if key in run:
            value = run[key]
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                if flag in cmd:
                    cmd[cmd.index(flag) + 1] = str(value)
                else:
                    cmd += [flag, str(value)]
    cmd += list(extra)
    return cmd


def sbatch_command(runs: Sequence[Dict], *, partition: str, mode: str, exclude: str = "") -> List[str]:
    array = ",".join(str(i) for i in range(len(runs)))
    cmd = ["sbatch", f"--partition={partition}", f"--array={array}"]
    gpu_mem = max(r.get("gpu_mem_gb", 12) for r in runs)
    if partition == "short":
        cmd.append("--gres=gpu:turing:1")                       # cu128 has no Pascal kernels
    else:
        cmd.append("--gres=gpu:1")
    if exclude:
        cmd.append(f"--exclude={exclude}")
    cmd += [f"--comment=recognize-{mode or 'full'}-{gpu_mem}gb", str(SBATCH)]
    return cmd


def _sinfo_nodes(partition: str) -> List[Tuple[str, str, str]]:
    """(node, gres, features) for every node of `partition`. Empty if sinfo is unavailable."""
    try:
        out = subprocess.run(["sinfo", "-p", partition, "-N", "-h", "-o", "%N|%G|%f"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            rows.append((parts[0].strip(), parts[1], parts[2]))
    return rows


def node_gpu_mem_gb(features: str) -> Optional[int]:
    """Largest per-GPU memory a node advertises, from its `gmemNN` features (None if it says nothing).

    The features are exact tags, not a ladder: one node may advertise only `gmem11` and another
    `gmem11,gmem12,gmem16`, so a `--constraint` on one tag is not a lower bound on GPU memory.
    Taking the max per node and excluding the nodes below the requirement is the reliable reading.
    """
    sizes = [int(m) for m in re.findall(r"\bgmem(\d+)\b", features)]
    return max(sizes) if sizes else None


def unusable_nodes(gpu_mem_gb: int, partition: str = "short,normal") -> str:
    """Nodes a run of this size must avoid, as a `--exclude` list.

    Two rules: torch 2.8+cu128 has no sm_61 kernels, so every Pascal node fails with `no kernel
    image is available`; and a node whose largest GPU is smaller than the run needs runs out of
    memory partway through. A node that advertises no `gmem` feature is unknown, so it is always
    excluded.
    """
    bad = set()
    for node, gres, features in _sinfo_nodes(partition):
        if "pascal" in gres:
            bad.add(node)
            continue
        have = node_gpu_mem_gb(features)
        if have is None or have < gpu_mem_gb:
            bad.add(node)
    return ",".join(sorted(bad))


def write_run_list(runs: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r["name"] + "\n" for r in runs))


def head_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
                         capture_output=True, text=True)
    return out.stdout.strip() or "?"


def assert_clean_tree() -> str:
    """Refuse to launch from a modified tree: a checkpoint's provenance must name code that exists.

    Every run stamps the commit it started at, so committing while runs are in flight (or launching
    with edits in the working tree) leaves checkpoints whose code cannot be recovered.
    """
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env).stdout.strip()
    if dirty:
        raise SystemExit("refusing to launch from a modified working tree; commit or stash first:\n" + dirty)
    return head_commit()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST,
                    help="run list to read (default: configs/runs.yaml, the campaign; "
                         "configs/runs_seeds.yaml holds training runs 1 and 2 of the two LoRA cells)")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    for name in ("command", "run"):
        p = sub.add_parser(name); p.add_argument("run_name"); p.add_argument("--smoke", action="store_true")
        p.add_argument("--timing", action="store_true")
    p = sub.add_parser("sbatch")
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--groups", nargs="+", default=None)
    p.add_argument("--partition", default="normal", choices=("normal", "short"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-dirty", action="store_true", help="launch even though the tree is modified (not for campaign runs)")
    args, extra = ap.parse_known_args(argv)
    extra = [a for a in extra if a != "--"]
    mode = "smoke" if getattr(args, "smoke", False) else ("timing" if getattr(args, "timing", False) else "")
    runs = load_runs(args.manifest)
    groups = sorted({r["group"] for r in runs})
    if getattr(args, "groups", None):
        unknown = sorted(set(args.groups) - set(groups))
        if unknown:
            raise SystemExit(f"unknown group(s) {unknown}; {args.manifest.name} has {groups}")

    if args.action == "list":
        print(f"{'#':>3}  {'name':<40} {'group':<9} {'gpu':>5}  deviations")
        for i, r in enumerate(runs):
            dev = ", ".join(f"{k}={r[k]}" for k in DEVIATIONS if k in r) or "-"
            print(f"{i:>3}  {r['name']:<40} {r['group']:<9} {r['gpu_mem_gb']:>3}GB  {dev}")
        print(f"{len(runs)} runs -> {CHECKPOINTS}")
        return 0

    if args.action in ("command", "run"):
        run = get_run(args.run_name, runs)
        cmd = build_command(run, mode=mode, extra=extra)
        if args.action == "command":
            print(" ".join(cmd))
            return 0
        env = dict(os.environ, PYTHONHASHSEED="0",
                   PYTHONPATH=f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts'}:{os.environ.get('PYTHONPATH', '')}")
        output_dir(run, mode).mkdir(parents=True, exist_ok=True)
        return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode

    selected = runs
    if args.groups:
        selected = [r for r in selected if r["group"] in args.groups]
    if args.runs:
        selected = [get_run(n, runs) for n in args.runs]
    if not selected:
        raise SystemExit("no runs selected")
    commit = head_commit() if getattr(args, "allow_dirty", False) else assert_clean_tree()
    selected = sorted(selected, key=lambda r: r["priority"])
    # One list file per submission: array tasks read it at execution time, so a later
    # submission must never overwrite the list an earlier array is still consuming.
    stamp_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    list_dir = CHECKPOINTS / (mode or "full")
    list_path = list_dir / f"runs_{stamp_id}.txt"
    suffix = 0
    while list_path.exists():                       # two submissions in the same second
        suffix += 1
        list_path = list_dir / f"runs_{stamp_id}-{suffix}.txt"
    need = max(r.get("gpu_mem_gb", 12) for r in selected)
    exclude = unusable_nodes(need, args.partition)
    nodes = {n for n, _, _ in _sinfo_nodes(args.partition)}
    if nodes and nodes <= set(exclude.split(",")):
        raise SystemExit(f"no node on `{args.partition}` has {need} GB and a supported GPU; "
                         f"split the submission or pick another partition")
    cmd = sbatch_command(selected, partition=args.partition, mode=mode, exclude=exclude)
    print(f"{len(selected)} runs at commit {commit} -> {list_path}")
    for r in selected:
        print(f"  {r['name']}")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    write_run_list(selected, list_path)
    print(f"RUN_LIST={list_path}")
    (CHECKPOINTS / (mode or "full") / "logs").mkdir(parents=True, exist_ok=True)
    # SUBMIT_COMMIT pins the campaign: a task that starts after HEAD has moved still records the
    # commit the campaign was launched from, so every run of one submission names the same code.
    env = dict(os.environ, RUN_LIST=str(list_path), RUN_MODE=mode, SUBMIT_COMMIT=commit,
               RUN_MANIFEST=str(args.manifest))
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
