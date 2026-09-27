"""Emit the paper's results tables from `results/`, so no number is typed by hand.

Two reporting choices shape the columns:

  P4 is reported as identity Rank-1, not exemplar mAP. Exemplar-level average precision with one
  relevant gallery entry per query is 1/rank, so it moves with the size of the gallery, which is
  the quantity gallery growth changes. A P4 mAP column would therefore not be comparable with a
  P2 mAP column.

  P2 is reported at Rank-1 beside P4 (in the cross-protocol table, as P4's static policy) so the
  P2-to-P4 comparison is like for like.

Cells are the mean over whatever training seeds exist for that configuration, up to three.
`--check` prints the per-seed spread instead of writing, so a cell that rests on one run is visible.

One table per `--table` value, written to `--out` (printed when `--out` is not given); the paths
are the files the paper inputs:

    --table cross      paper/generated/tables/popcharacters_eval_scores.tex
    --table p1grid     paper/generated/tables/p1_full_grid.tex
    --table p4decomp   paper/generated/tables/p4_decomposition.tex
    --table ablation   paper/generated/tables/memory_ablation.tex
    --table manga109   paper/generated/tables/manga109_eval_scores.tex, the Manga109 transfer grid
    --table permanga   paper/generated/tables/per_manga.tex, MagiV2 per test series
    --table cost       paper/generated/tables/cost.tex, the memory block's inference cost
                       (from results/latency.json)

    python paper/paper_tables.py --table cross \
        --out paper/generated/tables/popcharacters_eval_scores.tex
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _config import safe_name                     # noqa: E402
from recognize.data import load_split          # noqa: E402
from report import tables as T                    # noqa: E402

BACKBONES = [("transreid", "TransReID"), ("magiv2", "MagiV2"), ("magiv3", "MagiV3"),
             ("instructreid", "InstructReID"), ("reid5o", "ReID5o")]
MECHA = {"memory": "FT + Mem", "memory_lora": "FT + Mem + LoRA"}
SEEDS = (0, 1, 2)
B_MAX = "50"          # the protocol's operating point (src/recognize/protocol_constants.py)

# What a random ranking of the real gallery scores, measured by permutation over the actual
# protocol drivers rather than from a formula: `analysis/chance_baselines.py`,
# 8 test series x 5 gallery seeds x 200 draws. P1 and P2-R are that script's output. P2-T is the
# same procedure on the temporal seeding of `split_seeds`, which the script does not cover: the
# two seeding regimes are indistinguishable at this precision (a second random-seeding run gives
# 33.91 / 12.95 against 33.88 / 12.92, the permutation's own draw noise), so P2-T carries the
# P2-R values.
CHANCE = {"p1_map": 33.15, "p1_r1": 30.05,
          "p2r_map": 33.88, "p2r_r1": 12.92,
          "p2t_map": 33.88, "p2t_r1": 12.92}


def seeded(root: Path, series: List[str], bb: str, cfg: str, protocol: str,
           metric: str = "mAP", **kw) -> Dict[int, float]:
    """{training seed: macro over series}, over the seeds that are complete for this cell."""
    return {s: v * 100 for s, v in T.by_train_seed(root, series, bb, cfg, protocol, metric,
                                                   seeds=SEEDS, **kw).items()}


def mean(vals: Dict[int, float]) -> Optional[float]:
    return statistics.fmean(vals.values()) if vals else None


def fmt(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.1f}"


def best_mecha(root: Path, series: List[str], bb: str) -> str:
    """The memory-block configuration with the highest P1 mAP, on the same numbers reported."""
    scored = [(mean(seeded(root, series, bb, c, "p1")), c) for c in MECHA]
    scored = [(v, c) for v, c in scored if v is not None]
    return max(scored)[1]


def row(root: Path, series: List[str], bb: str, cfg: str) -> Dict:
    """One backbone-configuration line of the cross-protocol table.

    The P4 columns are the three update policies at the protocol's operating point, all at
    identity Rank-1. `frozen` is the static P2 gallery: it reproduces the P2 identity Rank-1
    column to the last decimal on every cell measured, which is why that column is labelled as
    the frozen policy instead of being printed twice.
    """
    def p4(strategy: str, policy: str) -> Optional[float]:
        return mean(seeded(root, series, bb, cfg, "p4", "R1_identity", strategy=strategy,
                           policy=policy, b_max=B_MAX))

    return {"p1_map":  mean(seeded(root, series, bb, cfg, "p1")),
            "p1_r1":   mean(seeded(root, series, bb, cfg, "p1", "R1")),
            "p2r_map": mean(seeded(root, series, bb, cfg, "p2", strategy="random")),
            "p2t_map": mean(seeded(root, series, bb, cfg, "p2", strategy="temporal")),
            "p4r_frz": p4("random", "frozen"),
            "p4r_prd": p4("random", "predicted"),
            "p4r_orc": p4("random", "oracle"),
            "p4t_frz": p4("temporal", "frozen"),
            "p4t_prd": p4("temporal", "predicted"),
            "p4t_orc": p4("temporal", "oracle"),
            "n_seeds": len(seeded(root, series, bb, cfg, "p1"))}


CROSS_COLS = ("p1_map", "p1_r1", "p2r_map", "p2t_map",
              "p4r_frz", "p4r_prd", "p4r_orc", "p4t_frz", "p4t_prd", "p4t_orc")


def render(root: Path, series: List[str]) -> str:
    chance = [CHANCE["p1_map"], CHANCE["p1_r1"], CHANCE["p2r_map"], CHANCE["p2t_map"],
              CHANCE["p2r_r1"], None, None, CHANCE["p2t_r1"], None, None]
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.0}",
           r"\caption{\textbf{Cross-protocol summary on \PopChars}, 8 held-out series, $k=1$ seed per "
           r"character, mean over three training runs. \emph{Chance} is a random ranking of the "
           r"same galleries. Per backbone the top row, \emph{Finetuned}, is the frozen backbone with a "
           r"trained BNNeck and no LoRA, and the bottom row ($\dagger$) the best memory-block configuration "
           r"by P1 mAP. Seeds are random (Seq-R) or first "
           r"appearances (Seq-T). P4 is identity Rank-1 at $B_{\max}=50$ under three policies: "
           r"\emph{static}, the P2 gallery; \emph{predicted}, adding each query under its top-1 match; "
           r"\emph{oracle}, adding it under its true character.}",
           r"\label{tab:cross_protocol}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{l|l|cc|cc|ccc|ccc}", r"\toprule", r"\rowcolor{gray!15}",
           r"& & \multicolumn{2}{c|}{\textbf{P1: Closed-Set}} "
           r"& \multicolumn{2}{c|}{\textbf{P2: Seeded}$_{k=1}$, mAP} "
           r"& \multicolumn{3}{c|}{\textbf{P4: Seq-R}$_{k=1}$, id.\ R-1} "
           r"& \multicolumn{3}{c}{\textbf{P4: Seq-T}$_{k=1}$, id.\ R-1} \\",
           r"\rowcolor{gray!15}",
           r"\multirow{-2}{*}{\textbf{Backbone}} & \multirow{-2}{*}{\textbf{Config}} "
           r"& \textbf{mAP} & \textbf{R-1} & \textbf{Seq-R} & \textbf{Seq-T} "
           r"& \textbf{Static} & \textbf{Pred.} & \textbf{Oracle} "
           r"& \textbf{Static} & \textbf{Pred.} & \textbf{Oracle} \\", r"\midrule",
           r"\emph{Chance} & Random ranking & " + " & ".join(fmt(v) for v in chance) + r" \\",
           r"\midrule"]
    for n, (bb, label) in enumerate(BACKBONES):
        best = best_mecha(root, series, bb)
        for cfg, name in (("finetuned", "Finetuned"), (best, MECHA[best] + r"$^\dagger$")):
            r = row(root, series, bb, cfg)
            head = label if cfg == "finetuned" else ""
            out.append(f"{head} & {name} & " + " & ".join(fmt(r[k]) for k in CROSS_COLS) + r" \\")
        out.append(r"\midrule" if n < len(BACKBONES) - 1 else r"\bottomrule")
    out += [r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(out) + "\n"


CONFIGS = [("pretrained", "Pretrained"), ("finetuned", "Finetuned"),
           ("memory", "Finetuned + Memory"), ("lora", "Finetuned + LoRA"),
           ("memory_lora", "Finetuned + Memory + LoRA")]


def _emph(vals: List[Optional[float]]) -> List[str]:
    """Bold the best and underline the second best of a column, skipping missing cells."""
    have = sorted([v for v in vals if v is not None], reverse=True)
    best = have[0] if have else None
    second = have[1] if len(have) > 1 else None
    out = []
    for v in vals:
        if v is None:
            out.append("--")
        elif v == best:
            out.append(f"\\textbf{{{v:.1f}}}")
        elif v == second:
            out.append(f"\\underline{{{v:.1f}}}")
        else:
            out.append(f"{v:.1f}")
    return out


def render_p1_grid(root: Path, series: List[str], suffix: str = "") -> str:
    """The full P1 closed-set grid, every backbone against every configuration."""
    metrics = [("mAP", "mAP"), ("R1", "R-1"), ("R5", "R-5"), ("R10", "R-10")]
    out = [r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.1}",
           r"\caption{Full P1 closed-set retrieval on the 8 held-out \PopChars series "
           r"(Appendix~\ref{app:reeval}). Best per backbone in \textbf{bold}, "
           r"second-best \underline{underlined}. \emph{Finetuned} trains a BNNeck on a frozen "
           r"backbone; only LoRA rows adapt backbone weights. Every row is the mean over three "
           r"training runs.}",
           r"\label{tab:suppl_p1_full}",
           r"\begin{tabular}{l|l|c|c|c|c}", r"\toprule", r"\rowcolor{gray!15}",
           r"\textbf{Backbone} & \textbf{Configuration} & " +
           " & ".join(f"\\textbf{{{lab}}}" for _, lab in metrics) + r" \\", r"\midrule"]
    for n, (bb, label) in enumerate(BACKBONES):
        cols = {k: [mean(seeded(root, series, bb, cfg, "p1", k, suffix=suffix))
                    for cfg, _ in CONFIGS] for k, _ in metrics}
        marked = {k: _emph(v) for k, v in cols.items()}
        out.append(f"\\multirow{{{len(CONFIGS)}}}{{*}}{{{label}}}")
        for r, (_, name) in enumerate(CONFIGS):
            out.append(f"  & {name:<26}& " + " & ".join(marked[k][r] for k, _ in metrics) + r" \\")
        out.append(r"\midrule" if n < len(BACKBONES) - 1 else r"\bottomrule")
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


PER_MANGA = [("finetuned", "Finetuned"), ("memory", "FT + Memory"), ("memory_lora", "FT + Mem + LoRA")]


def render_per_manga(root: Path, series: List[str], bb: str = "magiv2") -> str:
    """One backbone's P1 mAP and Rank-1 per test series, for the configurations around the headline.

    Each cell is the mean over the three training runs of that series' five-draw mean, so the macro
    of a column is the grid cell of Table `tab:suppl_p1_full`. Best per series in bold, ties at the
    printed precision bolded together.
    """
    import json

    def n_chars(name: str) -> int:
        d = json.loads((root / T.tag_for(bb, "finetuned", 0, "") / f"{safe_name(name)}.json").read_text())
        return int(d["series"]["n_identities"])

    label = dict(BACKBONES)[bb]
    out = [r"\begin{table}[!ht]",
           r"\centering \footnotesize \setlength{\tabcolsep}{5pt} \renewcommand{\arraystretch}{1.1} "
           r"\caption{\textbf{Per-manga P1 results} for " + label + r" across three configurations, mean "
           r"over three training runs and five seed draws. \#C is the character count. Best per series "
           r"in \textbf{bold}.} \label{tab:suppl_per_manga}",
           r"\begin{tabular}{l|c|cc|cc|cc}", r"\toprule", r"\rowcolor{gray!15}",
           r"& & " + " & ".join(
               (r"\multicolumn{2}{c|}{\textbf{%s}}" if i < len(PER_MANGA) - 1 else r"\multicolumn{2}{c}{\textbf{%s}}")
               % name for i, (_, name) in enumerate(PER_MANGA)) + r" \\",
           r"\rowcolor{gray!15}",
           r"\multirow{-2}{*}{\textbf{Manga}} & \multirow{-2}{*}{\textbf{\#C}} & "
           + " & ".join([r"\textbf{mAP} & \textbf{R-1}"] * len(PER_MANGA)) + r" \\", r"\midrule"]
    for name in series:
        cells = {}
        for metric in ("mAP", "R1"):
            vals = [mean(seeded(root, [name], bb, cfg, "p1", metric)) for cfg, _ in PER_MANGA]
            top = max(round(v, 1) for v in vals)
            cells[metric] = [(r"\textbf{%.1f}" if round(v, 1) == top else "%.1f") % v for v in vals]
        row_cells = [c for i in range(len(PER_MANGA)) for c in (cells["mAP"][i], cells["R1"][i])]
        out.append(f"{name} & {n_chars(name)} & " + " & ".join(row_cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def render_cost(path: Path) -> str:
    """The memory block's inference cost, from `analysis/latency.py`'s output."""
    import json
    d = json.loads(path.read_text())
    rows = {r["backbone"]: r for r in d["rows"]}
    gpu = d["device"].replace("NVIDIA ", "NVIDIA~")
    n = rows[BACKBONES[0][0]]["n_timed"]
    out = [r"\begin{table}[!ht]",
           r"\centering \footnotesize \setlength{\tabcolsep}{5pt} \renewcommand{\arraystretch}{1.1} "
           r"\caption{\textbf{Computational overhead} on one " + gpu + r" at batch size one in FP32, "
           r"median over " + str(n) + r" crops of one test series. \emph{$+$ Memory} is the two-pass "
           r"inference as implemented, which runs the backbone once per pass, and \emph{block alone} "
           r"is the same two passes over cached backbone features. Backbone parameters count what the "
           r"crop forward touches, since MagiV3 and ReID5o ship modules it never calls. The block's "
           r"buffers and prototypes are filled at test time and are not parameters.} \label{tab:suppl_cost}",
           r"\begin{tabular}{l|c|c|c|c|c}", r"\toprule", r"\rowcolor{gray!15}",
           r"& \multicolumn{3}{c|}{\textbf{Latency (ms/crop)}} & \multicolumn{2}{c}{\textbf{Parameters}} \\",
           r"\rowcolor{gray!15}",
           r"\multirow{-2}{*}{\textbf{Backbone}} & \textbf{Backbone} & \textbf{$+$ Memory} & "
           r"\textbf{Block alone} & \textbf{Backbone} & \textbf{Memory block} \\", r"\midrule"]
    for bb, label in BACKBONES:
        r = rows[bb]
        out.append(f"{label:<13}& {r['ms_backbone_median']:.1f} & {r['ms_memory_median']:.1f} & "
                   f"{r['ms_block_only_median']:.1f} & {r['params_backbone_used'] / 1e6:.1f}M & "
                   f"{r['params_memory_trainable'] / 1e6:.1f}M \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def render_manga109(root: Path, series: List[str]) -> str:
    """Manga109, zero-shot with \\PopChars-trained checkpoints."""
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.02}",
           r"\caption{\textbf{Cross-corpus transfer to Manga109}, 27 held-out volumes "
           r"(Appendix~\ref{app:reeval}). \emph{Pretrained} is the released backbone with no \PopChars training "
           r"at all. The two rows under it are \PopChars-trained checkpoints scored zero-shot "
           r"here, the no-memory baseline and then the best memory-block configuration by P1 mAP "
           r"($\dagger$), so the first gap in each block is what BNNeck fine-tuning on another "
           r"corpus is worth. P4 is identity Rank-1, with P2 given at Rank-1 beside it.}",
           r"\label{tab:manga109_results}", r"\resizebox{\linewidth}{!}{%",
           r"\begin{tabular}{l|l|cc|c|cc}", r"\toprule", r"\rowcolor{gray!15}",
           r"& & \multicolumn{2}{c|}{\textbf{P1: Closed-Set}} & \textbf{P2-R}$_{k=1}$ "
           r"& \multicolumn{2}{c}{\textbf{Seq-R}$_{k=1}$, id.\ R-1} \\", r"\rowcolor{gray!15}",
           r"\multirow{-2}{*}{\textbf{Backbone}} & \multirow{-2}{*}{\textbf{Config}} "
           r"& \textbf{mAP} & \textbf{R-1} & \textbf{mAP} & \textbf{P2} & \textbf{P4} \\", r"\midrule"]
    for n, (bb, label) in enumerate(BACKBONES):
        scored = [(mean(seeded(root, series, bb, c, "p1", suffix="__manga109")), c) for c in MECHA]
        best = max([(v, c) for v, c in scored if v is not None])[1]
        for cfg, name in (("pretrained", "Pretrained"), ("finetuned", "Finetuned"),
                          (best, MECHA[best] + r"$^\dagger$")):
            k = dict(suffix="__manga109")
            vals = [mean(seeded(root, series, bb, cfg, "p1", **k)),
                    mean(seeded(root, series, bb, cfg, "p1", "R1", **k)),
                    mean(seeded(root, series, bb, cfg, "p2", strategy="random", **k)),
                    mean(seeded(root, series, bb, cfg, "p2", "R1", strategy="random", **k)),
                    mean(seeded(root, series, bb, cfg, "p4", "R1_identity", strategy="random", **k))]
            out.append((label if cfg == "pretrained" else "") + f" & {name} & "
                       + " & ".join(fmt(v) for v in vals) + r" \\")
        out.append(r"\midrule" if n < len(BACKBONES) - 1 else r"\bottomrule")
    out += [r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(out) + "\n"


def render_p4_decomp(root: Path, series: List[str]) -> str:
    """P4 at k=1 decomposed into its three update policies, both seeding regimes.

    Identity Rank-1 at the protocol's operating point, because exemplar mAP with one relevant
    gallery entry per query is 1/rank and therefore moves with the size of the gallery, which is
    the quantity growth changes.
    """
    def cell(bb: str, cfg: str, strategy: str, policy: str, metric: str) -> Optional[float]:
        return mean(seeded(root, series, bb, cfg, "p4", metric, strategy=strategy,
                           policy=policy, b_max=B_MAX))

    def block(bb: str, cfg: str, strategy: str) -> List[Optional[float]]:
        return [cell(bb, cfg, strategy, p, "R1_identity") for p in ("frozen", "predicted", "oracle")] \
            + [cell(bb, cfg, strategy, "predicted", m) for m in ("wrong_append", "contamination")]

    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.02}",
           r"\caption{\textbf{P4 under its three update policies} at $k=1$ and $B_{\max}=50$ on the "
           r"8 held-out \PopChars test series, identity Rank-1, mean over three training runs and "
           r"five seed draws. \emph{Static} is the unchanged P2 gallery, \emph{predicted} adds each "
           r"query under its top-1 match (the protocol's rule) and \emph{oracle} adds each query "
           r"under its true character. \emph{Wrong-app.} is the fraction of added entries that carry "
           r"the wrong character and \emph{contam.} the mislabelled fraction of the grown gallery at "
           r"stream end, both under the predicted policy. \emph{Finetuned} trains a BNNeck head on "
           r"the frozen backbone, with no LoRA; \emph{FT + Mem} adds the memory block to it. The "
           r"three policies share one seed draw per run.}",
           r"\label{tab:p4_oracle}", r"\resizebox{\linewidth}{!}{%",
           r"\begin{tabular}{l|l|ccc|cc|ccc|cc}", r"\toprule", r"\rowcolor{gray!15}",
           r"& & \multicolumn{5}{c|}{\textbf{Seq-R} (random seeding)} "
           r"& \multicolumn{5}{c}{\textbf{Seq-T} (chronological seeding)} \\", r"\rowcolor{gray!15}",
           r"\multirow{-2}{*}{\textbf{Backbone}} & \multirow{-2}{*}{\textbf{Config}} "
           r"& \textbf{Static} & \textbf{Pred.} & \textbf{Oracle} & \textbf{Wrong-app.} "
           r"& \textbf{Contam.} & \textbf{Static} & \textbf{Pred.} & \textbf{Oracle} "
           r"& \textbf{Wrong-app.} & \textbf{Contam.} \\", r"\midrule"]
    for n, (bb, label) in enumerate(BACKBONES):
        for cfg, name in (("finetuned", "Finetuned"), ("memory", "FT + Mem")):
            vals = block(bb, cfg, "random") + block(bb, cfg, "temporal")
            head = label if cfg == "finetuned" else ""
            out.append(f"{head} & {name} & " + " & ".join(fmt(v) for v in vals) + r" \\")
        out.append(r"\midrule" if n < len(BACKBONES) - 1 else r"\bottomrule")
    out += [r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(out) + "\n"


ABLATIONS = [("no_wm", "working memory"), ("no_em", "episodic memory"),
             ("no_id_drop", "ID-drop"), ("no_mem_loss", "memory-consistency loss")]
ABLATION_BACKBONES = [("magiv2", "MagiV2"), ("magiv3", "MagiV3")]


def pairs(root: Path, series: List[str], bb: str, cfg: str, protocol: str,
          metric: str = "mAP", **kw) -> Dict:
    """{(series, training seed): value}, the paired unit the ablation verdict uses.

    8 series x 3 training seeds = 24 pairs, each the same configuration and its parent trained
    from the same seed and scored on the same series, which removes the training-seed spread.
    """
    import json
    out = {}
    for s in series:
        for seed in SEEDS:
            f = root / f"{bb}_{cfg}_seed{seed}" / f"{safe_name(s)}.json"
            if not f.exists():
                return {}
            c = T.cell([json.loads(f.read_text())], protocol, metric, **kw)
            if c is None:
                return {}
            out[(s, seed)] = c["mean"] * 100
    return out


def sign_test(delta: Dict, base: Dict) -> Dict:
    """Mean paired difference and the two-sided exact sign test over the shared pairs."""
    from scipy.stats import binomtest
    keys = sorted(set(delta) & set(base))
    d = [delta[k] - base[k] for k in keys]
    down = sum(1 for x in d if x < 0)
    return {"mean": statistics.fmean(d), "n": len(d), "n_down": down,
            "p": binomtest(down, len(d), 0.5).pvalue}


def render_ablation(root: Path, series: List[str]) -> str:
    """Per-component ablation of the memory block, on the two backbones whose memory works."""
    out = [r"\begin{table}[tb]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.02}",
           r"\caption{\textbf{Per-component ablation of the memory block} on the two backbones "
           r"where it earns anything, macro over the 8 held-out \PopChars test series over three "
           r"training runs. Each row removes one component from the full memory block; no row "
           r"carries LoRA. $\Delta$ is against the full memory block, paired over the 24 "
           r"(series, training run) cells, and $^{*}$ marks $p<0.05$ under a two-sided exact sign "
           r"test on those pairs. Working memory carries the whole effect: removing it returns P1 "
           r"mAP to the finetuned baseline on both backbones, while removing episodic memory, "
           r"ID-drop or the memory-consistency loss moves P1 mAP \emph{upward}.}",
           r"\label{tab:ablations}",
           r"\begin{tabular}{l|l|ccc|c}", r"\toprule", r"\rowcolor{gray!15}",
           r"\textbf{Backbone} & \textbf{Configuration} & \textbf{P1 mAP} & \textbf{P1 R-1} "
           r"& \textbf{P2-R@1 mAP} & \textbf{$\Delta$ P1 mAP vs full} \\", r"\midrule"]
    metrics = [("p1", "mAP", {}), ("p1", "R1", {}), ("p2", "mAP", {"strategy": "random", "k": "1"})]
    for n, (bb, label) in enumerate(ABLATION_BACKBONES):
        base = [pairs(root, series, bb, "memory", p, m, **kw) for p, m, kw in metrics]
        rows = [("finetuned", "Finetuned (no memory)"), ("memory", "Full memory block")] + \
               [(f"ablation_{a}", rf"\quad $-$ {lab}") for a, lab in ABLATIONS]
        for cfg, name in rows:
            vals = [pairs(root, series, bb, cfg, p, m, **kw) for p, m, kw in metrics]
            cells = [f"{statistics.fmean(v.values()):.2f}" if v else "--" for v in vals]
            if cfg == "memory":
                delta = "--"
            else:
                d = sign_test(vals[0], base[0])
                star = "^{*}" if d["p"] < 0.05 else ""
                delta = f"${'+' if d['mean'] >= 0 else '-'}{abs(d['mean']):.2f}{star}$"
            head = label if cfg == "finetuned" else ""
            out.append(f"{head} & {name} & " + " & ".join(cells) + f" & {delta}" + r" \\")
        out.append(r"\midrule" if n < len(ABLATION_BACKBONES) - 1 else r"\bottomrule")
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="results root; default results/popcharacters, or results/manga109 "
                         "for --table manga109 (the zero-shot Manga109 runs live in their own tree)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--table", default="cross",
                    choices=("cross", "p1grid", "manga109", "p4decomp", "ablation", "permanga", "cost"))
    ap.add_argument("--check", action="store_true", help="print per-seed spread, write nothing")
    args = ap.parse_args(argv)
    if args.results is None:
        args.results = ROOT / "results" / ("manga109" if args.table == "manga109" else "popcharacters")
    series = load_split()["test"]

    if args.check:
        print(f"{'backbone':<14}{'config':<14}{'seeds':>6}{'P1 mAP':>9}{'sd':>7}")
        for bb, _ in BACKBONES:
            for cfg in ("pretrained", "finetuned", "memory", "lora", "memory_lora"):
                v = seeded(args.results, series, bb, cfg, "p1")
                if not v:
                    continue
                sd = statistics.stdev(v.values()) if len(v) > 1 else 0.0
                print(f"{bb:<14}{cfg:<14}{len(v):>6}{mean(v):>9.2f}{sd:>7.3f}")
        return 0

    if args.table == "manga109":
        import yaml
        series = yaml.safe_load((ROOT / "configs/training/data_split_manga109.yaml").read_text())["val"]
        text = render_manga109(args.results, series)
    elif args.table == "p1grid":
        text = render_p1_grid(args.results, series)
    elif args.table == "permanga":
        text = render_per_manga(args.results, series)
    elif args.table == "cost":
        text = render_cost(ROOT / "results" / "latency.json")
    elif args.table == "p4decomp":
        text = render_p4_decomp(args.results, series)
    elif args.table == "ablation":
        text = render_ablation(args.results, series)
    else:
        text = render(args.results, series)
    if args.out is None:
        print(text)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[paper_tables] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
