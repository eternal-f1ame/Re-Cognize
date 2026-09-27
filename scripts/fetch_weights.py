#!/usr/bin/env python3
"""Fetch the released backbone checkpoints named in the registry and pin their SHA-256.

    python scripts/fetch_weights.py --all
    python scripts/fetch_weights.py transreid instructreid

Sources are tried in registry order ("gdrive:<id>" via gdown, or an https URL). A pinned
SHA-256 must match or the file is rejected; an unpinned one is printed and recorded in
reid_models/weights/SHA256SUMS so it can be pinned in the registry.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recognize.backbones import BACKBONE_REGISTRY, WEIGHTS_DIR  # noqa: E402

SUMS = WEIGHTS_DIR / "SHA256SUMS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(source: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith("gdrive:"):
        import gdown
        out = gdown.download(id=source[len("gdrive:"):], output=str(dest), quiet=False, fuzzy=True)
        if out is None:
            raise RuntimeError(f"gdown could not fetch {source}")
    else:
        urllib.request.urlretrieve(source, dest)


def record(path: Path, digest: str) -> None:
    SUMS.parent.mkdir(parents=True, exist_ok=True)
    lines = [l for l in SUMS.read_text().splitlines() if l.strip()] if SUMS.exists() else []
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    entry = f"{digest}  {rel}"
    if entry in lines:
        return                      # already recorded: leave the tracked file as it is
    lines = [entry if l.endswith("  " + rel) else l for l in lines]
    if entry not in lines:
        lines.append(entry)
    SUMS.write_text("\n".join(lines) + "\n")


def fetch(backbone: str, force: bool = False, downloader: Callable[[str, Path], None] = download) -> Optional[Path]:
    cfg = BACKBONE_REGISTRY[backbone]
    dest = cfg.weights_path()
    if dest is None:
        print(f"[fetch_weights] {backbone}: no released checkpoint file (HF repo {cfg.hf_repo}@{cfg.hf_revision})")
        return None
    if not dest.exists() or force:
        if not cfg.weights_sources:
            raise FileNotFoundError(f"{backbone}: {dest} is missing and the registry lists no download source")
        errors = []
        for src in cfg.weights_sources:
            try:
                print(f"[fetch_weights] {backbone}: downloading {src} -> {dest}")
                downloader(src, dest)
                break
            except Exception as e:  # noqa: BLE001 - try the next source, report all at the end
                errors.append(f"{src}: {type(e).__name__}: {e}")
                if dest.exists():
                    dest.unlink()
        else:
            raise RuntimeError(f"{backbone}: every source failed: " + "; ".join(errors))
    digest = sha256(dest)
    if cfg.weights_sha256 is not None and digest != cfg.weights_sha256:
        raise RuntimeError(f"{backbone}: SHA-256 mismatch for {dest}: got {digest}, registry pins {cfg.weights_sha256}")
    record(dest, digest)
    print(f"[fetch_weights] {backbone}: {dest.name} sha256 {digest[:16]}... {'(pinned)' if cfg.weights_sha256 else '(UNPINNED: add to registry)'}")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backbones", nargs="*", choices=sorted(BACKBONE_REGISTRY) + [[]], default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    names = sorted(BACKBONE_REGISTRY) if args.all or not args.backbones else args.backbones
    for name in names:
        fetch(name, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
