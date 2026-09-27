#!/usr/bin/env python3
"""Build the result tables from results/ JSON only.

    python scripts/report/tables.py [--corpus manga109] [--out results/reports/tables.md]

Every table is derived, never typed: each cell names the JSONs it came from through the
manifest of runs, and the commit that produced them, where the files record one, is printed in
the header.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _config import safe_name  # noqa: E402
from recognize.data import load_split  # noqa: E402
from recognize.protocol_constants import B_MAX  # noqa: E402

CONFIG_ORDER = ["pretrained", "pretrained__memory", "finetuned", "memory", "lora", "memory_lora"]
CONFIG_LABEL = {"pretrained": "Pretrained", "pretrained__memory": "Pretrained+Memory", "finetuned": "Finetuned",
                "memory": "Finetuned+Memory", "lora": "Finetuned+LoRA", "memory_lora": "Finetuned+Memory+LoRA"}
BACKBONES = ["transreid", "magiv2", "magiv3", "instructreid", "reid5o"]


def load_series(root: Path, tag: str, series: Iterable[str]) -> List[dict]:
    out = []
    for s in series:
        f = root / tag / f"{safe_name(s)}.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def cell(results: List[dict], protocol: str, metric: str = "mAP", *, strategy: str = "random",
         k: str = "1", policy: str = "predicted", b_max: str = str(B_MAX), rule: str = "fixed") -> Optional[Dict]:
    """Mean over seeds of the macro over series, with the sd across seeds."""
    per_seed: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        try:
            if protocol == "p1":
                for seed, m in r["p1"].items():
                    per_seed[seed].append(float(m[metric]))
            elif protocol == "p2":
                for seed, m in r["p2"][strategy][k].items():
                    per_seed[seed].append(float(m[metric]))
            elif protocol == "p4":
                for seed, m in r["p4"][strategy][k].items():
                    per_seed[seed].append(float(m[policy][b_max][metric]))
            elif protocol == "p3":
                per_seed["-"].append(float(r["p3"][rule][metric]))
        except (KeyError, TypeError):
            continue
    # `per_seed` is a defaultdict, so a KeyError raised while evaluating the value to append leaves
    # the key behind with an empty list, which is what happens when a result file lacks the metric.
    # Drop those before averaging rather than dividing by zero.
    per_seed = {k: v for k, v in per_seed.items() if v}
    if not per_seed:
        return None
    macros = [statistics.fmean(v) for v in per_seed.values()]
    return {"mean": statistics.fmean(macros),
            "sd": statistics.stdev(macros) if len(macros) > 1 else 0.0,
            "n_series": max(len(v) for v in per_seed.values()), "n_seeds": len(macros)}


def tag_for(backbone: str, config: str, seed: int = 0, suffix: str = "") -> str:
    if config.startswith("pretrained"):
        return f"pretrained__{backbone}" + ("__memory" if config.endswith("memory") else "") + suffix
    return f"{backbone}_{config}_seed{seed}" + suffix


def grid(root: Path, series: List[str], protocol: str, metric: str = "mAP", *,
         suffix: str = "", **kw) -> List[Dict]:
    rows = []
    for bb in BACKBONES:
        row = {"backbone": bb}
        for cfg in CONFIG_ORDER:
            c = cell(load_series(root, tag_for(bb, cfg, suffix=suffix), series), protocol, metric, **kw)
            row[cfg] = c
        rows.append(row)
    return rows


def render(rows: List[Dict], title: str, scale: float = 100.0, digits: int = 1,
           n_series: Optional[int] = None) -> str:
    head = f"### {title}\n\n| backbone | " + " | ".join(CONFIG_LABEL[c] for c in CONFIG_ORDER) + " |\n"
    head += "|---|" + "---|" * len(CONFIG_ORDER) + "\n"
    for row in rows:
        cells = []
        for cfg in CONFIG_ORDER:
            c = row[cfg]
            if c is None:
                cells.append("n/a")
                continue
            txt = f"{c['mean'] * scale:.{digits}f}"
            if c["n_seeds"] > 1:
                txt += f" ± {c['sd'] * scale:.{digits}f}"
            if n_series is not None and c["n_series"] < n_series:
                txt += f" ({c['n_series']}/{n_series} series)"      # partial evaluation, not a result
            cells.append(txt)
        head += f"| {row['backbone']} | " + " | ".join(cells) + " |\n"
    return head + "\n"


ABLATIONS = ["no_wm", "no_em", "no_id_drop", "no_mem_loss"]
ABLATION_LABEL = {"no_wm": "no working memory", "no_em": "no episodic memory",
                  "no_id_drop": "no ID-drop (rho=0)", "no_mem_loss": "no memory loss (w_mem=0)"}
TRAIN_SEEDS = (0, 1, 2)


def by_train_seed(root: Path, series: List[str], backbone: str, config: str, protocol: str,
                  metric: str = "mAP", seeds: Iterable[int] = TRAIN_SEEDS, *,
                  suffix: str = "", **kw) -> Dict[int, float]:
    """{training seed: macro over series, averaged over evaluation seeds}, for the seeds that exist.

    The sd this yields is across *training* seeds, which is the floor an architectural change has to
    clear. The sd inside `cell` is across evaluation seeds (gallery draws) and is a different, much
    smaller quantity; do not compare the two.
    """
    out = {}
    want = len(list(series))
    for s in seeds:
        c = cell(load_series(root, tag_for(backbone, config, s, suffix), series), protocol, metric, **kw)
        # a macro over four series is not comparable with a macro over eight, and mixing the two
        # inside an sd turns half-finished evaluation into what looks like run-to-run spread
        if c is not None and c["n_series"] == want:
            out[s] = c["mean"]
    return out


def _sd(values) -> Optional[float]:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else None


def _fmt(mean: Optional[float], sd: Optional[float] = None, scale: float = 100.0, digits: int = 2) -> str:
    if mean is None:
        return "n/a"
    return f"{mean * scale:.{digits}f}" + (f" ± {sd * scale:.{digits}f}" if sd is not None else "")


def render_seed_spread(root: Path, series: List[str], *, suffix: str = "") -> str:
    """Run-to-run spread of the whole pipeline: the same configuration trained at three seeds."""
    out = ["### Training-seed spread", "",
           "Each cell is one training run evaluated over all five gallery seeds and eight test series; the sd is across training seeds. This is the measurement floor: a difference between two configurations smaller than the sd of one configuration against itself means nothing.", "",
           "| backbone | config | n seeds | P1 mAP | P2-R@1 mAP | per-seed P1 |",
           "|---|---|---|---|---|---|"]
    floors = []
    for bb in BACKBONES:
        for cfg in ("finetuned", "memory", *(f"ablation_{a}" for a in ABLATIONS)):
            p1 = by_train_seed(root, series, bb, cfg, "p1", suffix=suffix)
            if len(p1) < 2:
                continue
            p2 = by_train_seed(root, series, bb, cfg, "p2", strategy="random", k="1", suffix=suffix)
            floors.append(statistics.stdev(p1.values()))
            per = ", ".join(f"s{k}={v * 100:.2f}" for k, v in sorted(p1.items()))
            label = CONFIG_LABEL.get(cfg) or f"Ablation: {ABLATION_LABEL[cfg[len('ablation_'):]]}"
            out.append(f"| {bb} | {label} | {len(p1)} | "
                       f"{_fmt(statistics.fmean(p1.values()), _sd(p1.values()))} | "
                       f"{_fmt(statistics.fmean(p2.values()) if p2 else None, _sd(p2.values()))} | {per} |")
    if floors:
        out += ["", f"Largest training-seed sd on P1 mAP: {max(floors) * 100:.2f} points, "
                    f"median {statistics.median(floors) * 100:.2f}, over {len(floors)} configurations. "
                    f"With three seeds the standard error of a configuration's mean is that sd over "
                    f"root 3, so a paired difference below about "
                    f"{2 * statistics.median(floors) * 100 / (3 ** 0.5):.2f} points is not resolvable."]
    return "\n".join(out) + "\n\n"


def render_ablations(root: Path, series: List[str], backbones: Iterable[str] = ("magiv2", "magiv3"),
                     *, suffix: str = "") -> str:
    """Memory ablations against the full memory model, paired by training seed."""
    out = ["### Memory ablations", "",
           "Each ablation is trained at the same three seeds as its parent, so the delta is paired: the mean of (ablation seed s - memory seed s) and the sd of those differences. A delta whose magnitude is below the training-seed sd in the table above is not a measurement.", "",
           "| backbone | ablation | n | P1 mAP | delta P1 | P2-R@1 mAP | delta P2-R@1 |",
           "|---|---|---|---|---|---|---|"]
    for bb in backbones:
        base1 = by_train_seed(root, series, bb, "memory", "p1", suffix=suffix)
        base2 = by_train_seed(root, series, bb, "memory", "p2", strategy="random", k="1", suffix=suffix)
        if not base1:
            continue
        out.append(f"| {bb} | (full memory) | {len(base1)} | "
                   f"{_fmt(statistics.fmean(base1.values()), _sd(base1.values()))} | -- | "
                   f"{_fmt(statistics.fmean(base2.values()) if base2 else None, _sd(base2.values()))} | -- |")
        for ab in ABLATIONS:
            a1 = by_train_seed(root, series, bb, f"ablation_{ab}", "p1", suffix=suffix)
            a2 = by_train_seed(root, series, bb, f"ablation_{ab}", "p2", strategy="random", k="1", suffix=suffix)
            if not a1:
                continue
            d1 = [a1[s] - base1[s] for s in sorted(a1) if s in base1]
            d2 = [a2[s] - base2[s] for s in sorted(a2) if s in base2]
            out.append(f"| {bb} | {ABLATION_LABEL[ab]} | {len(a1)} | "
                       f"{_fmt(statistics.fmean(a1.values()), _sd(a1.values()))} | "
                       f"{_fmt(statistics.fmean(d1) if d1 else None, _sd(d1))} | "
                       f"{_fmt(statistics.fmean(a2.values()) if a2 else None, _sd(a2.values()))} | "
                       f"{_fmt(statistics.fmean(d2) if d2 else None, _sd(d2))} |")
    return "\n".join(out) + "\n\n"


def default_paths(corpus: str) -> Tuple[Path, Path]:
    """The results tree and the output file for a corpus.

    Both follow from the corpus, so `--corpus manga109` on its own reads the Manga109 tree and
    writes its own file instead of overwriting the POPCharacters table with a grid of n/a.
    """
    stem = "tables" if corpus == "popcharacters" else f"tables_{corpus}"
    return ROOT / "results" / corpus, ROOT / "results" / "reports" / f"{stem}.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="default: results/<corpus>")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/reports/tables.md, or tables_<corpus>.md for a "
                         "corpus other than popcharacters")
    ap.add_argument("--corpus", default="popcharacters", choices=("popcharacters", "manga109"),
                    help="which corpus's test series the macro is taken over, and which tag suffix")
    args = ap.parse_args(argv)
    results, out = default_paths(args.corpus)
    args.results = args.results or results
    args.out = args.out or out
    if args.corpus == "popcharacters":
        series, suffix = load_split()["test"], ""
    else:
        from _config import DATASET_REGISTRY
        series = DATASET_REGISTRY[args.corpus].test_manga
        # the tag suffix travels as an argument; rebinding the module-level tag_for here would leak
        # the Manga109 naming into every later caller in the same process
        suffix = f"__{args.corpus}"
    n = len(series)
    if not args.results.exists():
        print(f"[tables] {args.results} does not exist yet; run the evaluation first")
        return 1
    # the commit is recorded by the harness for new runs; a result file without one is still read
    commits = {c for f in args.results.glob("*/*.json")
               if (c := json.loads(f.read_text())["provenance"].get("git_commit"))}
    at_commits = f" at commit(s) {', '.join(sorted(commits))}" if commits else ""
    title = {"popcharacters": "POPCharacters", "manga109": "Manga109, zero-shot"}[args.corpus]
    text = [f"# Results ({title} test series)", "",
            f"Series: {', '.join(series)}. Values are means over seeds of the macro over series, ± sd across seeds. Produced from {len(list(args.results.glob('*/*.json')))} result files{at_commits}.", ""]
    text.append(render(grid(args.results, series, "p1", suffix=suffix), "P1 closed set, mAP", n_series=n))
    text.append(render(grid(args.results, series, "p1", "R1", suffix=suffix), "P1 closed set, Rank-1", n_series=n))
    for strategy in ("random", "temporal"):
        label = "Seq-R" if strategy == "random" else "Seq-T"
        text.append(render(grid(args.results, series, "p2", strategy=strategy, suffix=suffix), f"P2 {label} at k=1, mAP", n_series=n))
        text.append(render(grid(args.results, series, "p4", strategy=strategy, metric="R1_identity", suffix=suffix),
                           f"P4 {label} at k=1, identity Rank-1 (B_max={B_MAX}, predicted)", n_series=n))
    text.append(render(grid(args.results, series, "p3", metric="purity", suffix=suffix), "P3 full stream, purity (fixed rule)", n_series=n))
    text.append(render(grid(args.results, series, "p3", metric="clusters", suffix=suffix), "P3 full stream, predicted clusters", scale=1.0, digits=0, n_series=n))
    text.append(render_seed_spread(args.results, series, suffix=suffix))
    text.append(render_ablations(args.results, series, suffix=suffix))
    body = "\n".join(text)
    # A table of nothing is a configuration error, not a result, and writing it would replace a
    # real table with an empty grid.
    # A measured cell carries a decimal or a "value ± sd"; a header cell like "P2-R@1 mAP" carries a
    # digit but is not a measurement, so match the shape rather than the presence of a digit.
    measured = re.compile(r"\d\.\d|\d\s*±")
    cells = [c for line in body.split("\n") if line.startswith("| ") for c in line.split("|")[2:-1]]
    if cells and not any(measured.search(c) for c in cells):
        print(f"[tables] every cell is n/a: no results for corpus {args.corpus!r} under "
              f"{args.results}. Nothing written.")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body)
    print(f"[tables] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
