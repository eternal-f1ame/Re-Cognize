#!/usr/bin/env python3
"""What each ingredient of the model is worth, on each of the four protocols.

Four elements, each a difference against the row it is meant to improve:

    fine-tuning    finetuned  - pretrained
    memory         memory     - finetuned
    LoRA           lora       - finetuned
    memory + LoRA  memory_lora- finetuned

The finetuned and memory cells have three training seeds on every backbone, so those differences are
paired by seed and carry a paired t-test. The two LoRA cells have one seed, so their differences are
single draws and are marked with a dagger; read them against the floor, not against a p-value.

P3 is not a like-for-like column. It runs on no-memory features (P3 has no seed images to
initialise the memory from), so the "memory" and "memory + LoRA" rows there compare two BNNecks
rather than memory on and off, and its two metrics move together for a bad reason: a run that
fragments the stream into more clusters scores higher purity. Both are shown so the pair can be
read together.

    python scripts/report/elements.py [--out results/reports/elements.md]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "report"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tables as T                                    # noqa: E402
from recognize.data import load_split              # noqa: E402
from recognize.protocol_constants import B_MAX        # noqa: E402

try:
    from scipy.stats import ttest_1samp, ttest_rel
except ImportError:                                   # scipy is a hard dependency of the repo
    ttest_rel = ttest_1samp = None

BACKBONES = ["transreid", "magiv2", "magiv3", "instructreid", "reid5o"]
ELEMENTS = [("fine-tuning", "pretrained", "finetuned"),
            ("memory", "finetuned", "memory"),
            ("LoRA", "finetuned", "lora"),
            ("memory + LoRA", "finetuned", "memory_lora")]
# (label, protocol, kwargs, scale, digits, higher-is-better)
METRICS = [("P1 mAP", "p1", {}, 100.0, 2, True),
           ("P1 Rank-1", "p1", {"metric": "R1"}, 100.0, 2, True),
           ("P2-R@1 mAP", "p2", {"strategy": "random", "k": "1"}, 100.0, 2, True),
           ("P2-T@1 mAP", "p2", {"strategy": "temporal", "k": "1"}, 100.0, 2, True),
           ("P3 ARI", "p3", {"metric": "ari"}, 100.0, 2, True),
           ("P3 purity", "p3", {"metric": "purity"}, 100.0, 2, True),
           ("P3 clusters", "p3", {"metric": "clusters"}, 1.0, 1, False),
           (f"P4-R@1 identity Rank-1", "p4",
            {"metric": "R1_identity", "strategy": "random", "k": "1"}, 100.0, 2, True)]
SEEDS = (0, 1, 2)


def values(root: Path, series: List[str], backbone: str, config: str,
           protocol: str, kw: Dict, scale: float) -> Dict[int, float]:
    """{training seed: macro over series}, only for seeds evaluated on every series."""
    if config.startswith("pretrained"):
        c = T.cell(T.load_series(root, T.tag_for(backbone, config), series), protocol,
                   kw.get("metric", "mAP"), **{k: v for k, v in kw.items() if k != "metric"})
        if c is None or c["n_series"] != len(series):
            return {}
        return {0: c["mean"] * scale}
    out = T.by_train_seed(root, series, backbone, config, protocol,
                          kw.get("metric", "mAP"), seeds=SEEDS,
                          **{k: v for k, v in kw.items() if k != "metric"})
    return {s: v * scale for s, v in out.items()}


def delta(before: Dict[int, float], after: Dict[int, float], *,
          before_deterministic: bool = False) -> Optional[Dict]:
    """The difference between two cells, using as much of the seed information as each side has.

    "paired"      both sides have the same three training seeds: mean of the paired differences,
                  with a paired t-test.
    "fixed"       the earlier side is the frozen pretrained backbone, which has no training seed at
                  all: the three seeds of the later side are each differenced against it, and a
                  one-sample t-test on those three differences is the right test.
    "one-seed"    one side is a single training run (the two LoRA cells). The difference uses the
                  other side's three-seed mean, which is the better estimate of that side, but the
                  single side's own spread is unmeasured, so no p-value is given.
    """
    if not before or not after:
        return None
    if before_deterministic and len(before) == 1 and len(after) > 1:
        base = next(iter(before.values()))
        d = [v - base for v in after.values()]
        p = float(ttest_1samp(d, 0.0).pvalue) if ttest_1samp and len(d) > 1 else None
        return {"delta": statistics.fmean(d), "n": len(d), "sd": statistics.stdev(d),
                "p": p, "kind": "fixed"}
    shared = sorted(set(before) & set(after))
    if len(shared) > 1:
        d = [after[s] - before[s] for s in shared]
        p = float(ttest_rel([after[s] for s in shared], [before[s] for s in shared]).pvalue) \
            if ttest_rel else None
        return {"delta": statistics.fmean(d), "n": len(d), "sd": statistics.stdev(d),
                "p": p, "kind": "paired"}
    return {"delta": statistics.fmean(after.values()) - statistics.fmean(before.values()),
            "n": min(len(before), len(after)), "sd": None, "p": None, "kind": "one-seed"}


MARK = {"paired": "", "fixed": "‡", "one-seed": "†"}


def fmt(d: Optional[Dict], digits: int) -> str:
    if d is None:
        return "n/a"
    txt = f"{d['delta']:+.{digits}f}"
    if d["p"] is not None and d["p"] < 0.05:
        txt = f"**{txt}**"
    return txt + MARK[d["kind"]]


def collect(root: Path, series: List[str]) -> Dict:
    """{(metric label, backbone, element label): delta record}."""
    out = {}
    for label, protocol, kw, scale, _digits, _hib in METRICS:
        for bb in BACKBONES:
            cache = {}
            for elem, before_cfg, after_cfg in ELEMENTS:
                for cfg in (before_cfg, after_cfg):
                    if cfg not in cache:
                        cache[cfg] = values(root, series, bb, cfg, protocol, kw, scale)
                out[(label, bb, elem)] = delta(cache[before_cfg], cache[after_cfg],
                                               before_deterministic=before_cfg.startswith("pretrained"))
    return out


def summary_table(rec: Dict) -> str:
    """Element x protocol: the mean over backbones, and how many of them move the right way."""
    head = ["### Every element on every protocol", "",
            "Mean over the five backbones, with the number of backbones the element helps on. "
            "For P3, fewer clusters is the improvement, so its count is of backbones where the "
            "element reduces fragmentation.", "",
            "| element | " + " | ".join(m[0] for m in METRICS) + " |",
            "|---|" + "---|" * len(METRICS)]
    for elem, _b, _a in ELEMENTS:
        cells = []
        for label, _p, _kw, _s, digits, hib in METRICS:
            ds = [rec[(label, bb, elem)] for bb in BACKBONES if rec.get((label, bb, elem))]
            if not ds:
                cells.append("n/a")
                continue
            mean = statistics.fmean(d["delta"] for d in ds)
            good = sum(1 for d in ds if (d["delta"] > 0) == hib)
            cells.append(f"{mean:+.{digits}f} ({good}/{len(ds)})")
        head.append(f"| {elem} | " + " | ".join(cells) + " |")
    return "\n".join(head) + "\n\n"


def detail_table(rec: Dict, label: str, digits: int) -> str:
    out = [f"| backbone | " + " | ".join(e[0] for e in ELEMENTS) + " |",
           "|---|" + "---|" * len(ELEMENTS)]
    for bb in BACKBONES:
        out.append(f"| {bb} | " + " | ".join(fmt(rec.get((label, bb, e[0])), digits)
                                             for e in ELEMENTS) + " |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "popcharacters")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "reports" / "elements.md")
    args = ap.parse_args(argv)
    series = load_split()["test"]
    if not args.results.exists():
        print(f"[elements] {args.results} does not exist yet")
        return 1
    rec = collect(args.results, series)

    text = ["# What each element is worth, protocol by protocol", "",
            f"POPCharacters, {len(series)} held-out series, the reported checkpoint (last epoch), "
            f"macro over series and mean over five gallery seeds. P4 is Seq-R at k=1 with "
            f"B_max={B_MAX} and predicted labels.", "",
            "**Bold** marks p < 0.05. An unmarked cell is a paired difference over the same three "
            "training seeds on both sides; ‡ differences the three seeds of the trained side "
            "against the frozen pretrained backbone, which has no training seed (a one-sample "
            "test); † is a cell where one side is a single training run, so no p-value is given "
            "and the number should be read against the floor. The floors, from 28 configurations "
            "at three seeds each, are P1 mAP 0.081, "
            "P1 Rank-1 0.303, P2-R@1 0.148, P2-T@1 0.474 and P4-R@1 identity Rank-1 0.512 for runs "
            "with memory, and three to seven times smaller without.", "",
            "P3 runs on no-memory features, so its memory rows compare two BNNecks rather than "
            "memory on and off. Read **ARI** as the P3 column: purity rises when a run fragments "
            "the stream into more clusters, so purity and cluster count have to be read together "
            "while ARI is not fooled by either. The true identity count is 4 to 12 per series "
            "against 27 to 279 predicted, so every model is over-fragmenting.", "",
            summary_table(rec)]
    for label, _p, _kw, _s, digits, _hib in METRICS:
        text.append(f"### {label}\n")
        text.append(detail_table(rec, label, digits))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(text))
    print(f"[elements] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
