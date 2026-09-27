"""Let a name in a dialogue bubble keep an identity live, instead of letting it append a crop.

Three mechanisms leave chronological seeding open (the EMA cast sheet of `prototype_gallery.py`, the recency-only live cast of `live_cast.py` and group pooling in `cluster_pool.py`), and all three read the appearance space that is already the problem. The only signal measured here that does not is a character's name spoken in a bubble, and the append break-even is the wrong bar for it.

The addressee rule of `dialogue_names.py` scores correct when the named character is among the *other* crops on the page. That is a statement of presence, not of identity, so it can never name a crop and is not comparable with a 40.9 % break-even *append* precision. What a presence signal fits is a candidate restriction, and a restriction has the opposite error profile: wrongly keeping an identity live only means the set shrank less, and only a wrong exclusion costs a query.

It is also the one anchor here that is not local. A name arrives on whatever page it falls, and locality is what limits every other anchor under chronological seeding. There the live cast's recall collapses because the live set is rebuilt from the system's own predictions, and those are worse exactly there. Names refresh it independently of the model, which breaks the loop the recency-only cast is caught in.

Arms hold the window fixed at the value selected on the dev series for the recency-only cast and vary only the name lifetime `L`, the number of pages a name keeps its identity live.

    python analysis/name_cast.py --device cuda --k 5 --strategy temporal \
        --names results/names.json --out results/namecast_t5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
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
from recognize.protocols import split_seeds                       # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
W = 20                       # the window selected on the dev series for the recency-only cast (live_cast.py)
LIFETIMES = (0, 1, 2, 5, 10, None)   # pages a name keeps an identity live; None = for ever


def run(features: np.ndarray, labels: np.ndarray, order: List[int],
        seed_map: Dict[int, List[int]], queries: List[int], gallery_only: Dict[int, List[int]],
        page_of: Dict[int, str], names_by_page: Dict[str, List[int]], *,
        window: Optional[int], lifetime: Optional[int], use_names: bool) -> Dict:
    """Stream in reading order; rank each query against the live identities only.

    The live set is the union of two sources. Recency contributes the identities the system itself named in the last `window` crops. Names contribute every identity a bubble named within the last `lifetime` pages, which needs no model and no locality. `use_names=False` reproduces the recency-only cast of `live_cast.py` (predicted source, exemplar gallery) exactly.
    """
    entries: Dict[int, List[int]] = defaultdict(list)
    for src in (seed_map, gallery_only):
        for c, idx in src.items():
            entries[int(c)].extend(idx)
    ids = sorted(entries)
    row = {c: r for r, c in enumerate(ids)}
    flat = [(c, i) for c in ids for i in entries[c]]
    bank = features[[i for _, i in flat]]
    owner = np.asarray([row[c] for c, _ in flat])

    pages_seen: List[str] = []
    for i in order:                                  # reading order over pages
        p = page_of[i]
        if not pages_seen or pages_seen[-1] != p:
            pages_seen.append(p)
    page_rank = {p: r for r, p in enumerate(pages_seen)}

    qset = set(queries)
    recent: deque = deque(maxlen=window)
    correct, static_ok, alive, n_live = [], [], [], []

    for pos in order:
        if pos not in qset:
            recent.append(int(labels[pos]))
            continue
        true = int(labels[pos])
        full = bank @ features[pos]
        static_ok.append(int(ids[owner[int(np.argmax(full))]]) == true)

        live = set(recent)
        if use_names:
            r0 = page_rank[page_of[pos]]
            for p, named in names_by_page.items():
                rp = page_rank.get(p)
                if rp is None or rp > r0:
                    continue                          # causal: only pages already read
                if lifetime is None or r0 - rp <= lifetime:
                    live.update(named)
        keep = {row[c] for c in live if c in row}
        if keep:
            mask = np.isin(owner, list(keep)); is_live = true in live
        else:
            mask = np.ones(len(bank), dtype=bool); is_live = True
        pred = int(ids[owner[mask][int(np.argmax(full[mask]))]])
        correct.append(pred == true)
        alive.append(is_live)
        n_live.append(len(keep) if keep else len(ids))
        recent.append(pred)

    c = np.asarray(correct); s = np.asarray(static_ok); L = np.asarray(alive)
    ell = float(L.mean()) if len(L) else 0.0
    return {"R1_identity": float(c.mean()) if len(c) else 0.0,
            "R1_static": float(s.mean()) if len(s) else 0.0,
            "measured_delta": float(c.mean() - s.mean()) if len(c) else 0.0,
            "ell": ell, "cast_size": float(np.mean(n_live)) if n_live else 0.0,
            "n_queries": len(c)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+",
                    default=["transreid", "magiv2", "magiv3", "instructreid", "reid5o"])
    ap.add_argument("--config", default="memory")
    ap.add_argument("--names", type=Path, default=Path("results/names.json"))
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", default="temporal", choices=("random", "temporal"))
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/namecast.json"))
    args = ap.parse_args(argv)

    names_raw = {r["series"]: r for r in json.loads(args.names.read_text())}
    series = ([n.strip() for n in args.series_file.read_text().split("\n") if n.strip()]
              if args.series_file else load_split()["test"])
    series = [s for s in series if s in names_raw]

    arms = {"static": None, f"live{W}/recency": dict(lifetime=0, use_names=False)}
    for L in LIFETIMES:
        arms[f"live{W}/names{'inf' if L is None else L}"] = dict(lifetime=L, use_names=True)

    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        ckpt = Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")
        if not ckpt.exists():
            print(f"[namecast] {ckpt} missing, skipping", flush=True); continue
        lm = load_checkpoint(str(ckpt), device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per_series: Dict = {}
        for name in series:
            stream = SeriesStream(args.data_root / name)
            labels, order = np.asarray(stream.labels), stream.reading_order
            page_of = {i: c.page_name for i, c in enumerate(stream.crops)}
            nb = {p: sorted(set(v["named"]))
                  for p, v in names_raw[name].get("named_on_page", {}).items()}
            tokens = FT.backbone_tokens(lm.model, stream, tf, batch_size=BATCH[bb], device=args.device)
            cell: Dict = {}
            for seed in args.seeds:
                sm, queries, go = split_seeds(labels, order, args.k, args.strategy, seed)
                g_idx = [i for v in sm.values() for i in v] + [i for v in go.values() for i in v]
                f = np.asarray(FT.extract(lm.model, stream, tf, mode="memory", gallery_idx=g_idx,
                                          batch_size=BATCH[bb], device=args.device,
                                          tokens=tokens).bn, dtype=np.float64)
                for tag, kw in arms.items():
                    if tag == "static":
                        kw2 = dict(window=None, lifetime=0, use_names=False)
                    else:
                        kw2 = dict(window=W, **kw)
                    cell.setdefault(tag, {})[str(seed)] = run(
                        f, labels, order, sm, queries, go, page_of, nb, **kw2)
            per_series[name] = cell
            print(f"[namecast] {bb} {name} done ({len(nb)} named pages)", flush=True)
        results[bb] = per_series
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\n[namecast] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
