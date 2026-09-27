"""Provenance stamping for Re:Cognize results.

Every result JSON should embed the output of `stamp()` so that reported
numbers can be traced back to the exact code revision, CLI arguments, seeds,
and checkpoint that produced them.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .protocol_constants import B_MAX, EVAL_SEEDS

# Repo root: src/recognize/provenance.py -> src/recognize -> src -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _run_git(args: list) -> Optional[str]:
    """Run a git subcommand in the repo root; return None on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            # GIT_OPTIONAL_LOCKS=0 keeps `git status` from refreshing (and locking) the index.
            # Many concurrent training runs stamp provenance, and one killed mid-status would
            # leave a stale .git/index.lock that blocks every later commit.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_commit() -> Optional[str]:
    return _run_git(["rev-parse", "--short", "HEAD"]) or None


#: Paths whose contents can change a result. Editing documentation while a campaign runs
#: does not mark its outputs as unreproducible.
CODE_PATHS = ("src", "scripts", "configs")


def _git_dirty(paths=CODE_PATHS) -> Optional[bool]:
    # --untracked-files=no makes porcelain ignore untracked files, so dirtiness only reflects
    # tracked (modified/staged) changes, and only under the paths that affect results.
    output = _run_git(["status", "--porcelain", "--untracked-files=no", "--", *paths])
    if output is None:
        return None
    return len(output) > 0


def _checkpoint_sha256(checkpoint_path: Optional[Union[str, Path]]) -> Optional[str]:
    if checkpoint_path is None:
        return None
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_config(
    checkpoint_path: Optional[Union[str, Path]]
) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        return None
    config = checkpoint["config"]
    if isinstance(config, dict):
        return config
    return vars(config)


def assert_hashseed_pinned() -> None:
    """Raise RuntimeError unless PYTHONHASHSEED is set to a non-empty value.

    Training and evaluation require a pinned PYTHONHASHSEED for reproducible runs;
    this should be checked at the top of every training/eval entry point.
    """
    if not os.environ.get("PYTHONHASHSEED"):
        raise RuntimeError(
            "PYTHONHASHSEED is not set. Training and evaluation require a pinned "
            "PYTHONHASHSEED (e.g. `export PYTHONHASHSEED=0`) for "
            "reproducible runs."
        )


def stamp(
    args: Union[argparse.Namespace, Dict[str, Any]],
    checkpoint_path: Optional[Union[str, Path]],
) -> Dict[str, Any]:
    """Build a provenance stamp for a training or evaluation run.

    Args:
        args: parsed CLI arguments, as an `argparse.Namespace` or a plain
            dict. Stored verbatim (as a dict) under the `args` key.
        checkpoint_path: path to a torch checkpoint whose identity should
            be recorded, or None if no checkpoint is involved.

    Returns:
        A dict with git, argument, seed, and checkpoint provenance
        fields suitable for embedding in a result JSON.
    """
    args_dict: Dict[str, Any] = (
        vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    )

    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),                       # code only; see CODE_PATHS
        "git_dirty_files": _run_git(["status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS]) or None,
        # The commit the campaign was launched from, when a launcher set it. `git_commit` is
        # whatever HEAD was when this process started, which drifts if HEAD moves mid-campaign.
        "submit_commit": os.environ.get("SUBMIT_COMMIT"),
        "args": args_dict,
        "seeds": args_dict.get("seeds", list(EVAL_SEEDS)),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "mask_ratio": args_dict.get("mask_ratio"),
        "b_max": args_dict.get("b_max", B_MAX),
        "split": args_dict.get("split"),
        "n_crops": args_dict.get("n_crops"),
        "n_identities": args_dict.get("n_identities"),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "checkpoint_config": _checkpoint_config(checkpoint_path),
    }
