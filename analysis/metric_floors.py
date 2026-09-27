"""The measurement floor of every metric we quote, from the training seeds of the reported checkpoint.

For every configuration with three evaluated training seeds, the sd over the seeds, all read at the
reported checkpoint (the last epoch): the spread of the pipeline against itself when nothing but
the seed changes.

    python analysis/metric_floors.py
"""
import sys, statistics
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/report"); sys.path.insert(0, "src")
import tables as T
from recognize.data import load_split
root = Path("results/popcharacters"); series = load_split()["test"]
BB = ("transreid", "magiv2", "magiv3", "instructreid", "reid5o")
METRICS = (("P1 mAP", "p1", {}), ("P1 R1", "p1", {"metric": "R1"}),
           ("P2-R@1 mAP", "p2", {}), ("P2-T@1 mAP", "p2", {"strategy": "temporal"}),
           ("P4-R@1 id-R1", "p4", {"metric": "R1_identity"}),
           ("P4-T@1 id-R1", "p4", {"strategy": "temporal", "metric": "R1_identity"}))
# every (backbone, config) that has three evaluated seeds, so the ablations join automatically
CONFIGS = ("finetuned", "memory", "lora", "memory_lora", "ablation_no_wm", "ablation_no_em",
           "ablation_no_id_drop", "ablation_no_mem_loss")
N_SERIES = len(series)


def complete(tag: str) -> bool:
    """A tag counts only when every test series is in: a macro over two series is not comparable
    with a macro over eight, and a half-written tag would masquerade as run-to-run spread."""
    return len(list((root / tag).glob("*.json"))) == N_SERIES


SEEDED = [(bb, cfg) for bb in BB for cfg in CONFIGS
          if all(complete(f"{bb}_{cfg}_seed{s}") for s in (0, 1, 2))]


def m(tag, protocol, **kw):
    c = T.cell(T.load_series(root, tag, series), protocol, **kw)
    return None if c is None else c["mean"] * 100


print(f"{'metric':<16}{'memory: median':>16}{'mean':>8}{'max':>8}"
      f"{'no memory: median':>19}{'max':>8}")
for label, protocol, kw in METRICS:
    sds = {"memory": [], "finetuned": []}
    for bb, cfg in SEEDED:
        vals = [m(f"{bb}_{cfg}_seed{s}", protocol, **kw) for s in (0, 1, 2)]
        vals = [v for v in vals if v is not None]
        if len(vals) == 3:
            sds["finetuned" if cfg == "finetuned" else "memory"].append(statistics.stdev(vals))
    def stat(v, f):
        return f(v) if v else float("nan")
    print(f"{label:<16}{stat(sds['memory'], statistics.median):>16.3f}"
          f"{stat(sds['memory'], statistics.fmean):>8.3f}{stat(sds['memory'], max):>8.3f}"
          f"{stat(sds['finetuned'], statistics.median):>19.3f}{stat(sds['finetuned'], max):>8.3f}")
print(f"\nEstimated from {len(SEEDED)} configurations with three complete seeds: "
      f"{sum(1 for c in SEEDED if c[1] != 'finetuned')} with memory, "
      f"{sum(1 for c in SEEDED if c[1] == 'finetuned')} without.")
print("Quote the median as the floor and the max as the worst case. Per-step reseeding of the")
print("training RNG makes one run reproducible given its seed; it does not shrink the spread *between* seeds.")
print("\nWith three seeds the standard error of a configuration's mean is the sd over root 3, and a")
print("paired difference of two configurations has a standard error of about the floor times")
print("root(2/3). Memory runs are the noisier group on every metric.")
