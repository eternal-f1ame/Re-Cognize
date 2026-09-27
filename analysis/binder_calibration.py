"""Can a backbone bind its own crops within a page, and at what threshold?

Two-stage transport is worth +12.5 to +16.9 identity Rank-1 under chronological seeding, but the binder that produces it is MagiV2's embedding. That one released model is also the panel detector, the page-affinity source behind must-link, the page groups behind the cast sheet, and one of the five backbones being scored, so it is an instrument of the framework and a subject of it at once.

A self-binder control, in which each backbone binds its own crops, needs a threshold per embedding space. A single cosine threshold of 0.7 applied to every backbone is not one setting, because the spaces are not comparable: same-identity cosine is 0.11 on TransReID against 0.09 on MagiV2, with different spreads. A threshold calibrated for one space says nothing in another, so within-page must-link precision has to be measured for each backbone.

Within-page grouping is also a far easier task than the one these backbones score 20 % on. Two crops on one page share art style, scene, lighting and often a panel. A backbone that is weak across a 600-crop volume may still be strong at "are these two crops on this page the same character", and if it is, the binder is free and the dependency on one model dissolves.

This measures must-link precision and coverage for every backbone over a threshold sweep, so each can be calibrated on a labelled slice the way the commit condition's terms are.

    python analysis/binder_calibration.py --device cuda --out results/binder_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
TAUS = [round(0.05 * i, 2) for i in range(2, 20)]                  # 0.10 .. 0.95


def page_clusters(feats: np.ndarray, page_of: List[str], tau: float) -> List[Optional[str]]:
    """Single-linkage within each page at cosine >= tau. Returns a cluster id per crop."""
    by_page: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(page_of):
        by_page[p].append(i)
    out: List[Optional[str]] = [None] * len(page_of)
    for p, idx in by_page.items():
        if len(idx) == 1:
            out[idx[0]] = f"{p}#0"
            continue
        F = feats[idx]
        S = F @ F.T
        parent = list(range(len(idx)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if S[a, b] >= tau:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        for a in range(len(idx)):
            out[idx[a]] = f"{p}#{find(a)}"
    return out


def quality(cl: List[Optional[str]], labels) -> Dict:
    """Must-link precision over within-cluster pairs, and how many pairs the rule asserts."""
    mem: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(cl):
        if c:
            mem[c].append(i)
    ok = n = 0
    for _, idx in mem.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ok += int(labels[idx[a]] == labels[idx[b]]); n += 1
    grouped = sum(len(v) for v in mem.values() if len(v) > 1)
    return {"must_link": ok / n if n else None, "pairs": n,
            "grouped_share": grouped / max(len(cl), 1), "clusters": len(mem)}


def self_test() -> None:
    """Two pages, known answer."""
    f = np.array([[1, 0], [1, 0], [0, 1],      # page A: crops 0,1 identical; 2 orthogonal
                  [1, 0], [0, 1]], dtype=float)  # page B: crops 3,4 orthogonal
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    page = ["A", "A", "A", "B", "B"]
    labels = [7, 7, 9, 7, 9]
    cl = page_clusters(f, page, 0.9)
    assert cl[0] == cl[1] and cl[0] != cl[2], "identical crops on a page must group, orthogonal must not"
    assert cl[3] != cl[4], "orthogonal crops on page B must not group"
    q = quality(cl, labels)
    assert q["pairs"] == 1 and q["must_link"] == 1.0, f"one asserted pair, correct: {q}"
    # a threshold below every similarity groups a whole page, including a wrong pair
    cl2 = page_clusters(f, page, -1.0)
    q2 = quality(cl2, labels)
    assert q2["pairs"] == 3 + 1 and q2["must_link"] < 1.0, f"loose tau must assert wrong pairs: {q2}"
    print("binder_calibration self-test: ok")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+",
                    default=["transreid", "instructreid", "reid5o", "magiv3", "magiv2"])
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/binder_calibration.json"))
    args = ap.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    self_test()

    series = ([s.strip() for s in args.series_file.read_text().split("\n") if s.strip()]
              if args.series_file else load_split()["test"])
    out: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ck = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ck.exists():
            print(f"[binder] {ck} missing, skipping", flush=True); continue
        lm = load_checkpoint(str(ck), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per: Dict = {}
        for name in series:
            st = SeriesStream(args.data_root / name)
            labels = np.asarray(st.labels)
            page = [c.page_name for c in st.crops]
            f = np.asarray(FT.extract(lm.model, st, tf, mode="none", batch_size=BATCH[bb],
                                      device=args.device).bn, dtype=np.float64)
            per[name] = {f"{t:.2f}": quality(page_clusters(f, page, t), labels) for t in args.taus}
            print(f"[binder] {bb} {name} done", flush=True)
        out[bb] = per
        del lm
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"\n[binder] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
