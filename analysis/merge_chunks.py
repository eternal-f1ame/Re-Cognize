#!/usr/bin/env python3
"""Merge analysis outputs that were run in chunks of series into one file.

Long runs over the 27 Manga109 volumes are split into three series lists
(results/series/m109_chunk{0,1,2}.lst) and run per backbone. Each output maps
backbone -> series -> result, so the merged file is the union of the per-series entries:

    python analysis/merge_chunks.py results/proto_m109_k5_magiv2c{0,1,2}.json \\
        results/proto_m109_k5_magiv3c{0,1,2}.json --out results/proto_m109_k5.json

A series that appears in two inputs with different results is an error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def merge(paths: List[Path]) -> Dict:
    out: Dict[str, Dict] = {}
    for p in paths:
        for backbone, per_series in json.loads(p.read_text()).items():
            into = out.setdefault(backbone, {})
            for series, result in per_series.items():
                if series in into and into[series] != result:
                    raise ValueError(f"{p}: {backbone}/{series} differs from an earlier input")
                into[series] = result
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    merged = merge(args.inputs)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"[merge_chunks] {', '.join(f'{b}: {len(s)} series' for b, s in merged.items())} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
