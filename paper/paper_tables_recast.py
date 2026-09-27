"""Emit the paper's Re:Cast, transfer, commit-condition and binding tables from `results/`.

Sibling of `paper_tables.py`, which writes the POPCharacters grid, the full P1 grid and the
Manga109 table, and like it types no number by hand. This module writes four tables:

  --table recast       -> paper/generated/tables/recast_combined.tex   (tab:recast_combined)
                          Re:Cast one change at a time, with seed expansion beside it
  --table crosscorpus  -> paper/generated/tables/cross_corpus_summary.tex (tab:cross_corpus)
                          the headline rows of the appendix Manga109 and Re:Verse tables
  --table commit       -> paper/generated/tables/commit_condition.tex   (tab:commit)
                          the commit-condition terms, from results/commit_*.json
                          (`analysis/commit_condition.py`)
  --table transport    -> paper/generated/tables/transport_results.tex (tab:transport)
                          two-stage binding under chronological seeding, from
                          results/transport_stage1_self.json, transport.json,
                          transport_m109_*.json and exact_*.json
                          (`analysis/transport_append.py`)

Sources of the first two, all under `results/` and all at identity Rank-1 for P4:

  proto_k1.json, proto_k5.json, proto_other_k5.json   prototype-gallery sweep
                                                      (`analysis/prototype_gallery.py`)
  expand_k1.json, expand_k5.json                      seed-expansion experiment
                                                      (`analysis/seed_expansion.py`)
  proto_m109_k5.json                                  the same sweep on 27 Manga109 volumes
  manga109/<tag>/<volume>.json, reverse/<tag>/Re-Zero.json   output of `scripts/evaluate.py`

Two bases are in play and the caption says so. The prototype sweep scores every arm on the
protocol's own query set. The seed-expansion experiment removes the expanded crops from the
query set of every arm, its own static reference included, so its static sits below the
prototype sweep's. Cells from the two are never subtracted from each other.

The arms are the settings of one axis: `exemplar/frozen` is the static bag of exemplars,
`proto/frozen/mean5+anchor` the cast sheet, `proto/mustlink/mean5+anchor` the cast sheet plus
panel-linked commitment, `recast+expand` that plus seed expansion, `proto/oracle/mean5+anchor`
the same representation grown under true labels.

    python paper/paper_tables_recast.py --table recast \
        --out paper/generated/tables/recast_combined.tex
    python paper/paper_tables_recast.py --table crosscorpus \
        --out paper/generated/tables/cross_corpus_summary.tex
    python paper/paper_tables_recast.py --table recast --check   # numbers, no LaTeX
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from recognize.data import load_split          # noqa: E402
from report import tables as T                    # noqa: E402

BACKBONES = [("transreid", "TransReID"), ("magiv2", "MagiV2"), ("magiv3", "MagiV3"),
             ("instructreid", "InstructReID"), ("reid5o", "ReID5o")]
GALLERY_SEEDS = ("0", "1", "2")

STATIC_PROTO = "exemplar/frozen"
STATIC_EXPAND = "exemplar/static"
CAST = "proto/frozen/mean5+anchor"
COMMIT = "proto/mustlink/mean5+anchor"
ORACLE = "proto/oracle/mean5+anchor"
EXPAND = "recast+expand"
RECAST_IN_EXPAND = "recast"
METRIC = "R1_identity"


# --------------------------------------------------------------------------- diagnostics I/O

def load_json(*names: str) -> Dict:
    """Merge the per-backbone diagnostic files into one {backbone: {series: {arm: ...}}}."""
    out: Dict = {}
    for n in names:
        for bb, per_series in json.loads((ROOT / "results" / n).read_text()).items():
            out.setdefault(bb, {}).update(per_series)
    return out


def per_series(data: Dict, bb: str, arm: str) -> Optional[Dict[str, float]]:
    """{series: mean over gallery seeds}, or None when this backbone has no such arm."""
    if bb not in data:
        return None
    cells = data[bb]
    if arm not in next(iter(cells.values())):
        return None
    return {s: statistics.fmean(c[arm][t][METRIC] for t in GALLERY_SEEDS) for s, c in cells.items()}


def macro(data: Dict, bb: str, arm: str) -> Optional[float]:
    v = per_series(data, bb, arm)
    return None if v is None else 100 * statistics.fmean(v.values())


def delta(data: Dict, bb: str, arm: str, base: str) -> Optional[Tuple[float, float, int, int]]:
    """(mean difference in points, paired p over series, series improved, series)."""
    a, b = per_series(data, bb, arm), per_series(data, bb, base)
    if a is None or b is None:
        return None
    keys = sorted(a)
    diffs = [100 * (a[k] - b[k]) for k in keys]
    if max(abs(d) for d in diffs) < 1e-12:            # an arm identical to its baseline
        return 0.0, 1.0, 0, len(diffs)
    p = float(stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys]).pvalue)
    return statistics.fmean(diffs), p, sum(1 for d in diffs if d > 0), len(diffs)


def fmt_delta(d: Optional[Tuple[float, float, int, int]], *, bold: bool = True) -> str:
    if d is None:
        return "---"
    v, p, _, _ = d
    s = f"{v:+.2f}"
    return f"\\textbf{{{s}}}" if bold and p < 0.05 else s


def fmt_abs(v: Optional[float]) -> str:
    return "---" if v is None else f"{v:.2f}"


# --------------------------------------------------------------------------- table 1: Re:Cast

CAPTION_RECAST = (
    r"\textbf{\ReCast, one change at a time}: P4 identity Rank-1 with random seeds at "
    r"$B_{\max}=50$, on the 8 held-out \PopChars series and, zero-shot, on 27 held-out Manga109 "
    r"volumes. Each column after \emph{Static} adds one change and gives the cumulative gain over "
    r"\emph{Static}, except \emph{$+$ expansion}, which is scored against its own static reference "
    r"with the expanded crops removed from the queries; \emph{Oracle} adds every query under its "
    r"true label. Cells are one training run and three seed draws, and \textbf{bold} marks $p<0.05$ "
    r"on a paired $t$-test over series. At $k=1$ the first two changes coincide, the oracle is in "
    r"Table~\ref{tab:cross_protocol}, and seed expansion was not run on Manga109."
)


def render_recast() -> str:
    pk1 = load_json("proto_k1.json")
    pk5 = load_json("proto_k5.json", "proto_other_k5.json")
    ek1, ek5 = load_json("expand_k1.json"), load_json("expand_k5.json")
    m109 = load_json("proto_m109_k5.json")

    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.0}",
           r"\caption{" + CAPTION_RECAST + "}", r"\label{tab:recast_combined}",
           r"\begin{tabular}{l|c|cc|c|c}", r"\toprule", r"\rowcolor{gray!15}",
           r"\textbf{Backbone} & \textbf{Static} & \textbf{Cast sheet} & \textbf{$+$ commitment} "
           r"& \textbf{$+$ expansion} & \textbf{Oracle} \\", r"\midrule",
           r"\multicolumn{6}{l}{\PopChars\textit{, 8 test series, Seq-R, $k=1$}} \\"]

    for bb, label in BACKBONES:                      # k=1: the seed-expansion basis throughout
        joint = delta(ek1, bb, RECAST_IN_EXPAND, STATIC_EXPAND)
        out.append(f"{label} & {fmt_abs(macro(ek1, bb, STATIC_EXPAND))} & "
                   + r"\multicolumn{2}{c|}{" + fmt_delta(joint) + "} & "
                   + fmt_delta(delta(ek1, bb, EXPAND, STATIC_EXPAND)) + r" & --- \\")

    out += [r"\midrule",
            r"\multicolumn{6}{l}{\PopChars\textit{, 8 test series, Seq-R, $k=5$}} \\"]
    for bb, label in BACKBONES:
        out.append(f"{label} & {fmt_abs(macro(pk5, bb, STATIC_PROTO))} & "
                   + fmt_delta(delta(pk5, bb, CAST, STATIC_PROTO)) + " & "
                   + fmt_delta(delta(pk5, bb, COMMIT, STATIC_PROTO)) + " & "
                   + fmt_delta(delta(ek5, bb, EXPAND, STATIC_EXPAND)) + " & "
                   + fmt_delta(delta(pk5, bb, ORACLE, STATIC_PROTO), bold=False) + r" \\")

    out += [r"\midrule",
            r"\multicolumn{6}{l}{\textit{Manga109, 27 held-out volumes, zero-shot, Seq-R, $k=5$}} \\"]
    for bb, label in BACKBONES:
        if bb not in m109:
            continue
        out.append(f"{label} & {fmt_abs(macro(m109, bb, STATIC_PROTO))} & "
                   + fmt_delta(delta(m109, bb, CAST, STATIC_PROTO)) + " & "
                   + fmt_delta(delta(m109, bb, COMMIT, STATIC_PROTO)) + r" & --- & "
                   + fmt_delta(delta(m109, bb, ORACLE, STATIC_PROTO), bold=False) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def check_recast() -> None:
    pk1 = load_json("proto_k1.json")
    pk5 = load_json("proto_k5.json", "proto_other_k5.json")
    ek1, ek5 = load_json("expand_k1.json"), load_json("expand_k5.json")
    m109 = load_json("proto_m109_k5.json")
    for tag, data, base in (("pop k=1 (expand basis)", ek1, STATIC_EXPAND),
                            ("pop k=5 (proto basis)", pk5, STATIC_PROTO),
                            ("manga109 k=5", m109, STATIC_PROTO)):
        print(f"--- {tag}")
        for bb, label in BACKBONES:
            if bb not in data:
                continue
            arms = ([("joint", RECAST_IN_EXPAND)] if data is ek1 else
                    [("cast", CAST), ("commit", COMMIT), ("oracle", ORACLE)])
            line = f"  {label:<13}static {macro(data, bb, base):6.2f}"
            for name, arm in arms:
                d = delta(data, bb, arm, base)
                line += f" | {name} {d[0]:+6.2f} p={d[1]:.4f} {d[2]}/{d[3]}" if d else f" | {name} ---"
            exp = delta(ek1 if data is ek1 else ek5, bb, EXPAND, STATIC_EXPAND) \
                if data is not m109 else None
            if exp:
                line += f" | expand {exp[0]:+6.2f} p={exp[1]:.4f} {exp[2]}/{exp[3]}"
            print(line)
    print("--- k=1 prototype sweep, the two backbones it covers")
    for bb in ("magiv2", "magiv3"):
        for name, arm in (("cast", CAST), ("commit", COMMIT), ("oracle", ORACLE)):
            d = delta(pk1, bb, arm, STATIC_PROTO)
            print(f"  {bb:<9}static {macro(pk1, bb, STATIC_PROTO):6.2f} {name:<7}"
                  f"{d[0]:+6.2f} p={d[1]:.4f}")
    print("--- expansion static below the prototype static (points)")
    for tag, e, p in (("k=5", ek5, pk5),):
        for bb, label in BACKBONES:
            print(f"  {tag} {label:<13}{macro(p, bb, STATIC_PROTO) - macro(e, bb, STATIC_EXPAND):.2f}")


# ------------------------------------------------------------------- table 2: cross-corpus

CAPTION_CROSS = (
    r"\textbf{Transfer beyond \PopChars}, headline rows. Manga109 is 27 held-out volumes. "
    r"\emph{Pretrained} is the released backbone with no \PopChars training at all; "
    r"\emph{Finetuned} and \emph{$+$ Mem.} are \PopChars-trained checkpoints scored zero-shot "
    r"here from one training run, the latter the memory-block configuration with the highest P1 "
    r"mAP on this corpus, so the first gap is what BNNeck fine-tuning on another corpus is worth "
    r"and the second is the memory block. P4 is identity Rank-1 at $k=1$ under random seeding at "
    r"$B_{\max}=50$, for that same configuration. Re:Verse is one series (Re:Zero) and its two "
    r"columns are the released backbone with no \PopChars training. Manga109 cells are the mean "
    r"over three seed draws, Re:Verse cells over five. The full tables are "
    r"Table~\ref{tab:manga109_results} and Table~\ref{tab:reverse_results} in the appendix. "
    r"Three vision-language models on the same Re:Verse benchmark reach 1.11, 0.00 and 0.85 "
    r"percent character-identification accuracy~\citep{baranwal2025reverse}."
)


MECHA = {"memory": "FT + Mem", "memory_lora": "FT + Mem + LoRA"}


def m109_cell(series: List[str], bb: str, cfg: str, protocol: str, metric: str = "mAP",
              **kw) -> Optional[float]:
    tag = T.tag_for(bb, cfg, 0, "__manga109")
    c = T.cell(T.load_series(ROOT / "results" / "manga109", tag, series), protocol, metric, **kw)
    return None if c is None else 100 * c["mean"]


def reverse_cell(bb: str, metric: str) -> Optional[float]:
    f = ROOT / "results" / "reverse" / f"pretrained__{bb}__reverse" / "Re-Zero.json"
    if not f.exists():
        return None
    p1 = json.loads(f.read_text())["p1"]
    return 100 * statistics.fmean(float(v[metric]) for v in p1.values())


def render_crosscorpus() -> str:
    series = yaml.safe_load((ROOT / "configs/training/data_split_manga109.yaml").read_text())["val"]
    rows = []
    for bb, label in BACKBONES:
        best = max((m109_cell(series, bb, c, "p1"), c) for c in MECHA)[1]
        rows.append((label, [m109_cell(series, bb, "pretrained", "p1"),
                             m109_cell(series, bb, "finetuned", "p1"),
                             m109_cell(series, bb, best, "p1"),
                             m109_cell(series, bb, best, "p4", "R1_identity"),
                             reverse_cell(bb, "mAP"), reverse_cell(bb, "R1")]))
    best_in = [max(r[1][c] for r in rows) for c in range(6)]

    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.05}",
           r"\caption{" + CAPTION_CROSS + "}", r"\label{tab:cross_corpus}",
           r"\begin{tabular}{l|ccc|c|cc}", r"\toprule", r"\rowcolor{gray!15}",
           r"& \multicolumn{4}{c|}{\textbf{Manga109}, 27 volumes} "
           r"& \multicolumn{2}{c}{\textbf{Re:Verse}, 1 series, pretrained} \\", r"\rowcolor{gray!15}",
           r"& \multicolumn{3}{c|}{P1 mAP} & \textbf{P4 id.\ R-1} "
           r"& \multicolumn{2}{c}{P1} \\", r"\rowcolor{gray!15}",
           r"\multirow{-3}{*}{\textbf{Backbone}} & \textbf{Pretrained} & \textbf{Finetuned} "
           r"& \textbf{$+$ Mem.} & \textbf{$+$ Mem.} & \textbf{mAP} & \textbf{R-1} \\", r"\midrule"]
    for label, vals in rows:
        cells = [("---" if v is None else
                  (f"\\textbf{{{v:.1f}}}" if v == best_in[i] else f"{v:.1f}"))
                 for i, v in enumerate(vals)]
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def check_crosscorpus() -> None:
    series = yaml.safe_load((ROOT / "configs/training/data_split_manga109.yaml").read_text())["val"]
    print(f"manga109 volumes: {len(series)}")
    for bb, label in BACKBONES:
        best = max((m109_cell(series, bb, c, "p1"), c) for c in MECHA)[1]
        print(f"  {label:<13} best={best:<12} "
              f"P1 pre {m109_cell(series, bb, 'pretrained', 'p1'):.2f} "
              f"P1 ft {m109_cell(series, bb, 'finetuned', 'p1'):.2f} "
              f"P1 best {m109_cell(series, bb, best, 'p1'):.2f} "
              f"P4 best {m109_cell(series, bb, best, 'p4', 'R1_identity'):.2f} "
              f"| reverse pre {reverse_cell(bb, 'mAP'):.2f} / {reverse_cell(bb, 'R1'):.2f}")


# ------------------------------------------------------------- table 3: the commit condition

# Every record on disk (one series x one gallery seed) carries the exact identity
# Delta_i = c_i (p_i - a_i+). A row has to print terms that reproduce its own prediction, so the
# aggregation is fixed by the algebra rather than chosen: c is the plain mean of c_i, while p_eff
# and a+ are capture-weighted, sum(c_i x_i) / sum(c_i). Then
#     c_bar (p_tilde - a_tilde+) = (1/N) sum c_i p_i - (1/N) sum c_i a_i+ = mean(Delta_i)
# identically. Taking plain means of p_eff and a+ instead opens a Jensen gap: it moves every row by
# 0.07 to 0.17 and flips the sign of MagiV2 / POPCharacters, the paper's worked exception, from
# the measured -0.04 to +0.33. `a` is not a term of the identity and stays a plain mean.
CC_ROWS = [("\\PopChars", "TransReID", "commit_pop_transreid.json", "transreid"),
           ("\\PopChars", "InstructReID", "commit_pop_instructreid.json", "instructreid"),
           ("\\PopChars", "ReID5o", "commit_pop_reid5o.json", "reid5o"),
           ("\\PopChars", "MagiV3", "commit_pop_magiv3.json", "magiv3"),
           ("Manga109", "MagiV3", "commit_m109_magiv3.json", "magiv3"),
           ("\\PopChars", "MagiV2", "commit_pop_magiv2.json", "magiv2"),
           ("Manga109", "MagiV2", "commit_m109_magiv2.json", "magiv2")]

CAPTION_COMMIT = (
    r"\textbf{The commit condition reproduces every measured change.} Commitment under the page "
    r"constraint (Section~\ref{sec:recast}) on the bag of $k=5$ random seeds per character, without "
    r"the cast sheet, over three seed draws. $a$ is the static gallery's identity Rank-1 and $c$, "
    r"$p_{\mathrm{eff}}$ and $a^{+}$ are the terms of Equation~\ref{eq:commit}, in percent. "
    r"Weighting $p_{\mathrm{eff}}$ and $a^{+}$ by each record's capture rate, "
    r"$\sum_i c_i x_i/\sum_i c_i$, makes $c\,(p_{\mathrm{eff}}-a^{+})$ exactly the mean measured "
    r"change, so the last two columns agree."
)


def commit_rows() -> List[Dict]:
    """One dict per printed row, with the terms aggregated so the identity survives printing."""
    rows = []
    for corpus, label, fname, bb in CC_ROWS:
        cell = json.loads((ROOT / "results" / fname).read_text())[bb]
        recs = [r for per in cell["by_series"].values() for r in per["mustlink"]]
        w = sum(r["c"] for r in recs)
        rows.append({"corpus": corpus, "label": label, "backbone": bb, "n_records": len(recs),
                     "a": 100 * statistics.fmean(r["a"] for r in recs),
                     "c": 100 * statistics.fmean(r["c"] for r in recs),
                     "p_eff": 100 * sum(r["c"] * r["p_eff"] for r in recs) / w,
                     "a_plus": 100 * sum(r["c"] * r["a_plus"] for r in recs) / w,
                     "predicted": 100 * statistics.fmean(r["predicted"] for r in recs),
                     "measured": 100 * statistics.fmean(r["delta"] for r in recs)})
    return rows


def commit_check(rows: List[Dict]) -> List[str]:
    """Fail loudly unless the *printed* terms rebuild the printed prediction, and it matches."""
    bad = []
    for r in rows:
        recon = round(r["c"], 2) / 100 * (round(r["p_eff"], 3) - round(r["a_plus"], 3))
        if f"{recon:+.2f}" != f"{r['predicted']:+.2f}":
            bad.append(f"{r['corpus']} {r['label']}: printed terms give {recon:+.2f}, "
                       f"prediction prints {r['predicted']:+.2f}")
        if f"{r['predicted']:+.2f}" != f"{r['measured']:+.2f}":
            bad.append(f"{r['corpus']} {r['label']}: prediction {r['predicted']:+.4f} against "
                       f"measured {r['measured']:+.4f}")
    return bad


def render_commit() -> str:
    rows = commit_rows()
    bad = commit_check(rows)
    if bad:
        raise SystemExit("[paper_tables_recast] commit table does not close:\n  " + "\n  ".join(bad))
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.0}",
           r"\caption{" + CAPTION_COMMIT + "}", r"\label{tab:commit}",
           r"\begin{tabular}{ll|cccc|cc}", r"\toprule", r"\rowcolor{gray!15}",
           r"\textbf{Corpus} & \textbf{Backbone} & $\boldsymbol{a}$ & $\boldsymbol{c}$ & "
           r"$\boldsymbol{p_{\mathrm{eff}}}$ & $\boldsymbol{a^{+}}$ & "
           r"$\boldsymbol{c(p_{\mathrm{eff}}\!-\!a^{+})}$ & \textbf{measured} \\", r"\midrule"]
    for r in rows:
        out.append(f"{r['corpus']} & {r['label']} & {r['a']:.2f} & {r['c']:.2f}\\% & "
                   f"{r['p_eff']:.3f} & {r['a_plus']:.3f} & "
                   f"{r['predicted']:+.2f} & {r['measured']:+.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def check_commit() -> None:
    rows = commit_rows()
    print(f"{'corpus':<15}{'backbone':<14}{'n':>4}{'a':>8}{'c':>9}{'p_eff':>10}{'a+':>10}"
          f"{'c(p-a+)':>10}{'measured':>10}{'resid':>11}")
    for r in rows:
        print(f"{r['corpus'].replace(chr(92) + 'PopChars', 'POPCharacters'):<15}{r['backbone']:<14}"
              f"{r['n_records']:>4}{r['a']:>8.2f}{r['c']:>8.2f}%{r['p_eff']:>10.3f}"
              f"{r['a_plus']:>10.3f}{r['predicted']:>+10.2f}{r['measured']:>+10.2f}"
              f"{r['predicted'] - r['measured']:>+11.2e}")
    bad = commit_check(rows)
    print("printed terms reproduce the printed prediction on every row" if not bad
          else "MISMATCH:\n  " + "\n  ".join(bad))
    ps = [r["p_eff"] for r in rows if r["corpus"] == r"\PopChars"]
    print(f"p_eff across the 5 POPCharacters cells: {min(ps):.2f} to {max(ps):.2f}; "
          f"across all 7: {min(r['p_eff'] for r in rows):.2f} to "
          f"{max(r['p_eff'] for r in rows):.2f}")


# ----------------------------------------------------------------- table 4: two-stage binding

# Chronological-seeding transport. Every column is a difference in P4 identity Rank-1 from the
# same static gallery ("none") of the same file, averaged over series and gallery seeds. Every
# file below is output of analysis/transport_append.py.
#
#   shared binder, POPCharacters   transport_stage1_self.json      (arm "transport", temporal)
#   shared binder, Manga109        transport_m109_<bb>.json
#   self binder,   POPCharacters   exact_<bb>.json at SELF_TAU[bb]
#   oracle / shuffled              transport_stage1_self.json      (POPCharacters, same file as a)
#   Seq-R control                  transport.json         (random regime)
#
# SELF_TAU is the threshold the commit condition selects: the one maximising the predicted
# c (p_eff - a) over the self-binder threshold sweep of analysis/transport_append.py
# (--link-source self --stage1 self --taus 0.35 0.45 0.55 0.65 0.75 0.85 --strategies temporal).
# It is NOT the threshold that maximises the measured gain; that sweep is the same file's other
# columns, and selecting on it would be selecting on the test metric. The selection measured c
# as appends per query, an upper bound on the capture rate that transport_append.py reports, so
# selecting with the capture rate can choose differently: on MagiV2 the measured gain at 0.75
# (+10.99) exceeds the one at the selected 0.55 (+8.46).
SELF_TAU = {"transreid": "0.35", "instructreid": "0.35", "reid5o": "0.35",
            "magiv3": "0.55", "magiv2": "0.55"}
TRANSPORT_BACKBONES = [("transreid", "TransReID"), ("instructreid", "InstructReID"),
                       ("reid5o", "ReID5o"), ("magiv3", "MagiV3"), ("magiv2", "MagiV2")]


def _transport_series(node: Dict, arm: str) -> Dict[str, float]:
    """{series: mean over gallery seeds} for one arm of one {series: {arm: {seed: ...}}} map."""
    return {s: statistics.fmean(c[arm][t][METRIC] for t in sorted(c[arm]))
            for s, c in node.items()}


def _transport_node(name: str, bb: str, regime: str, tau: Optional[str] = None) -> Dict:
    raw = json.loads((ROOT / "results" / name).read_text())[bb]
    return {s: (rd[regime] if tau is None else rd[regime][tau]) for s, rd in raw.items()}


def _transport_delta(node: Dict, arm: str) -> Tuple[float, float, int, int]:
    a, b = _transport_series(node, arm), _transport_series(node, "none")
    keys = sorted(a)
    diffs = [100 * (a[k] - b[k]) for k in keys]
    p = float(stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys]).pvalue)
    return statistics.fmean(diffs), p, sum(1 for d in diffs if d > 0), len(diffs)


def transport_rows() -> List[Dict]:
    rows = []
    for bb, label in TRANSPORT_BACKBONES:
        pop = _transport_node("transport_stage1_self.json", bb, "temporal")
        m109 = _transport_node(f"transport_m109_{bb}.json", bb, "temporal")
        self_ = _transport_node(f"exact_{bb}.json", bb, "temporal", SELF_TAU[bb])
        seqr = _transport_node("transport.json", bb, "random")
        rows.append({
            "backbone": bb, "label": label,
            "a": 100 * statistics.fmean(_transport_series(pop, "none").values()),
            "shared_pop": _transport_delta(pop, "transport"),
            "shared_m109": _transport_delta(m109, "transport"),
            "self": _transport_delta(self_, "transport"),
            "self_tau": SELF_TAU[bb],
            "oracle_pop": _transport_delta(pop, "oracle"),
            "oracle_m109": _transport_delta(m109, "oracle"),
            "shuffled_pop": _transport_delta(pop, "shuffled"),
            "shuffled_m109": _transport_delta(m109, "shuffled"),
            "seqr": _transport_delta(seqr, "transport"),
        })
    return rows


CAPTION_TRANSPORT = (
    r"\textbf{Two-stage binding under chronological seeding}: change in P4 identity Rank-1 over the "
    r"static gallery ($a$ on \PopChars) at $k=5$, averaged over series and seed draws; \textbf{bold} is a gain at $p<0.05$ paired over series. Columns are \PopChars\ "
    r"unless marked Manga109 (27 volumes, zero-shot). \emph{Shared} is the binder of "
    r"Section~\ref{sec:binding} for every backbone and \emph{Self} each backbone's own. "
    r"\emph{Oracle} is the same representation's ceiling ($+20.2$ to $+26.4$ on Manga109), "
    r"\emph{Shuffled} permutes the added identities ($-8.1$ to $-43.5$ on Manga109), and "
    r"\emph{Seq-R} the same binding with random seeds."
)


def render_transport() -> str:
    rows = transport_rows()
    out = [r"\begin{table}[t]", r"\centering \footnotesize \setlength{\tabcolsep}{3pt} "
           r"\renewcommand{\arraystretch}{1.0}",
           r"\caption{" + CAPTION_TRANSPORT + "}", r"\label{tab:transport}",
           r"\begin{tabular}{l|c|cc|c|c|cc}", r"\toprule", r"\rowcolor{gray!15}",
           r"& & \multicolumn{2}{c|}{\textbf{Shared binder}} & \textbf{Self} & & "
           r"\multicolumn{2}{c}{\textbf{Controls}} \\", r"\rowcolor{gray!15}",
           r"\multirow{-2}{*}{\textbf{Backbone}} & \textbf{$a$} & \textbf{\PopChars} & "
           r"\textbf{Manga109} & \textbf{\PopChars} & \textbf{Oracle} & \textbf{Shuffled} & "
           r"\textbf{Seq-R} \\", r"\midrule"]
    for r in rows:
        out.append(f"{r['label']:<13} & {r['a']:.1f} & {fmt_delta(r['shared_pop'])} & "
                   f"{fmt_delta(r['shared_m109'], bold=r['shared_m109'][0] > 0)} & "
                   f"{fmt_delta(r['self'])} & "
                   f"{fmt_delta(r['oracle_pop'], bold=False)} & "
                   f"{fmt_delta(r['shuffled_pop'], bold=False)} & "
                   f"{fmt_delta(r['seqr'], bold=False)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def check_transport() -> None:
    rows = transport_rows()
    print(f"{'backbone':<14}{'a':>7}{'shared pop':>13}{'shared m109':>14}{'self':>12}"
          f"{'oracle pop':>13}{'shuf pop':>11}{'shuf m109':>12}{'seq-R':>11}")
    for r in rows:
        def cell(d):
            return f"{d[0]:+.2f} (p={d[1]:.3f})"
        print(f"{r['label']:<14}{r['a']:>7.1f}{cell(r['shared_pop']):>13}"
              f"{cell(r['shared_m109']):>14}{cell(r['self']):>12}"
              f"{cell(r['oracle_pop']):>13}{cell(r['shuffled_pop']):>11}"
              f"{cell(r['shuffled_m109']):>12}{cell(r['seqr']):>11}")
    pop = [r["shared_pop"][0] for r in rows]
    m109 = [r["shared_m109"][0] for r in rows if r["shared_m109"][0] > 0]
    print(f"shared binder, POPCharacters: {min(pop):+.2f} to {max(pop):+.2f}; "
          f"share of the POPCharacters oracle "
          f"{min(r['shared_pop'][0] / r['oracle_pop'][0] for r in rows) * 100:.0f} to "
          f"{max(r['shared_pop'][0] / r['oracle_pop'][0] for r in rows) * 100:.0f}%")
    print(f"shared binder, Manga109, the four it helps: {min(m109):+.2f} to {max(m109):+.2f}")
    print(f"self binder at the condition's threshold: "
          f"{min(r['self'][0] for r in rows):+.2f} to {max(r['self'][0] for r in rows):+.2f}")
    print(f"shuffled, POPCharacters: {min(r['shuffled_pop'][0] for r in rows):+.2f} to "
          f"{max(r['shuffled_pop'][0] for r in rows):+.2f}; Manga109: "
          f"{min(r['shuffled_m109'][0] for r in rows):+.2f} to "
          f"{max(r['shuffled_m109'][0] for r in rows):+.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="recast",
                    choices=("recast", "crosscorpus", "commit", "transport"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="print the numbers, write nothing")
    args = ap.parse_args(argv)

    checks = {"recast": check_recast, "crosscorpus": check_crosscorpus,
              "commit": check_commit, "transport": check_transport}
    renders = {"recast": render_recast, "crosscorpus": render_crosscorpus,
               "commit": render_commit, "transport": render_transport}
    if args.check:
        checks[args.table]()
        return 0
    text = renders[args.table]()
    if args.out is None:
        print(text)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[paper_tables_recast] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
