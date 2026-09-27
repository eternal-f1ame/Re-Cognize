#!/usr/bin/env python3
"""Download the paper's per-tuple results and unpack them into results/.

Every table, figure and quoted number in the paper is computed from these files by the scripts in
analysis/ and paper/, so with them the paper's tables regenerate without a GPU:

    python scripts/fetch_results.py                                      # download, verify, unpack
    python scripts/fetch_results.py --archive recognize-results.tar.xz   # a local copy

The archive's SHA-256 is pinned below; a mismatch is an error, not a warning.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = "recognize-results.tar.xz"
SHA256 = "71510bf302743913c191be130195b8346d8b523e4e73032cf8d383d6cfab3112"
# The archive, attached to the repository's neurips-2026 release.
URL = "https://github.com/eternal-f1ame/Re-Cognize/releases/download/neurips-2026/recognize-results.tar.xz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(archive: Path, dest: Path, expected: str = SHA256) -> int:
    """Verify `archive` against `expected` and extract it under `dest`. Returns the file count."""
    digest = sha256(archive)
    if digest != expected:
        raise RuntimeError(f"SHA-256 mismatch for {archive}: got {digest}, expected {expected}")
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        bad = [m.name for m in members if not m.name.startswith("results/")]
        if bad:
            raise RuntimeError(f"{archive} holds files outside results/: {bad[:3]}")
        tar.extractall(dest, members=members, filter="data")
    return len(members)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=None, help="use a local copy instead of downloading")
    ap.add_argument("--dest", type=Path, default=ROOT, help="directory that receives results/ (default: repo root)")
    args = ap.parse_args(argv)

    archive = args.archive
    if archive is None:
        archive = ROOT / ARCHIVE
        if not archive.exists():
            print(f"[fetch_results] downloading {URL}")
            urllib.request.urlretrieve(URL, archive)
    n = unpack(archive, args.dest)
    print(f"[fetch_results] {n} files -> {args.dest / 'results'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
