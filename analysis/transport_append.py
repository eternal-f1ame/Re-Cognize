"""Carry an identity forward through an online crop-to-crop merge, and score it as a gallery.

Two-stage binding (`stage2_merge.py`) gives 62.5 % cross-page must-link at median span 0.153, and a merged group that contains a seed transports that seed's identity to its other members at 54.1 % precision, reaching 27.5 % of queries at span 0.191. Every break-even under chronological seeding is between 19.7 and 36.7 %, so the relation clears all of them. Those figures do not give identity Rank-1, and the commit condition decides that: Delta = c (p_eff - a_plus), of which the offline merge establishes only p_eff.

**The merge here is causal; the one in `stage2_merge.py` is not.** `stage2_merge.py` clusters a whole volume at once, so it decides about page 3 while looking at page 80. That is a ceiling, not a mechanism. This processes pages in reading order and lets a page cluster merge only into clusters built from pages already read, which is what P4 permits. Expect it to be worse than the offline figure, and the gap between them is the price of causality.

A cluster acquires a label when a seed joins it, and every later member is appended under that label. Seeds are protected and FIFO eviction applies to the grown portion, as in `run_p4`.

Arms: no growth, transport, shuffled (the same appends under permuted identities), append by top-1, and oracle. Random seeding is the control: there the gallery already holds a nearby reference, so transport should buy little however precise it is, and a gain there would mean the mechanism is not doing what the distance law says it does.

    python analysis/transport_append.py --device cuda --panels results/panels_o2o --out results/transport.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream, load_split            # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402
from recognize.protocols import split_seeds                       # noqa: E402
from _gallery import Gallery                                      # noqa: E402

BATCH = {"transreid": 128, "instructreid": 128, "reid5o": 64, "magiv2": 64, "magiv3": 16}
B_MAX = 50


def transport_labels(link_feats, page_clusters, order, pos, seed_of, tau):
    """Online merge in reading order. Returns crop -> identity a merged cluster would assign it.

    A page cluster is matched against clusters built from earlier pages only. The best match above
    `tau` absorbs it, otherwise it starts a new cluster. A cluster takes a label the first time a
    seed joins it, and keeps it.
    """
    # page clusters in reading order of their earliest crop
    cl_members: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(page_clusters):
        if c:
            cl_members[c].append(i)
    keys = sorted(cl_members, key=lambda k: min(pos[i] for i in cl_members[k]))

    means: List[np.ndarray] = []
    sizes: List[float] = []
    label: List[Optional[int]] = []
    out: Dict[int, int] = {}
    for k in keys:
        mem = cl_members[k]
        v = link_feats[mem].mean(0)
        v = v / max(np.linalg.norm(v), 1e-12)
        best, bi = tau, None
        for j in range(len(means)):
            s = float(means[j] @ v)
            if s > best:
                best, bi = s, j
        own = [seed_of[i] for i in mem if i in seed_of]
        if bi is None:
            means.append(v); sizes.append(float(len(mem)))
            label.append(own[0] if own else None)
        else:
            w = sizes[bi] + len(mem)
            m = (means[bi] * sizes[bi] + v * len(mem)) / w
            means[bi] = m / max(np.linalg.norm(m), 1e-12); sizes[bi] = w
            if label[bi] is None and own:
                label[bi] = own[0]
            bi_lab = label[bi]
            if bi_lab is not None:
                for i in mem:
                    if i not in seed_of:
                        out[i] = int(bi_lab)
    return out



def page_clusters_self(feats, crops, tau):
    """Stage 1 from the evaluated backbone's own features: single-linkage inside each page.

    The default stage 1 is MagiV2's character clustering, so a self-merge that keeps it still carries
    MagiV2 knowledge. This removes the teacher from both stages, which is the only version that tests
    whether two-stage transport is a principle rather than a distillation of one model.
    """
    by_page = defaultdict(list)
    for i, c in enumerate(crops):
        by_page[c.page_name].append(i)
    out = [None] * len(crops)
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


def run(feats, labels, order, k, strategy, seed, *, mode, transport=None, oracle=False):
    """One causal pass. Returns identity Rank-1 and the exact terms of the commit condition.

    `shuffled` appends the same crops at the same positions under permuted identities. If the gain
    survives it, the mechanism is adding exemplars rather than transporting identity.

    c, p_eff and a_plus are the equation's own terms, not proxies. A query is *captured* when the
    entry it matches is one growth added, which is what `Gallery.rank` reports. Appends per query is
    not the same quantity and overstates it, because an appended crop only captures a query when it
    wins top-1.
    """
    ids = sorted(set(int(x) for x in labels))
    rng = np.random.default_rng(9000 + seed)
    sh = list(ids); rng.shuffle(sh)
    perm = {a: b for a, b in zip(ids, sh)}

    seed_map, queries, gallery_only = split_seeds(labels, order, k, strategy, seed)
    seed_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
    pos = {c: t for t, c in enumerate(order)}
    T = max(1, len(order) - 1)
    g = Gallery(feats, labels, seed_idx, b_max=B_MAX)

    refs: Dict[int, List[int]] = defaultdict(list)
    for i in sorted(seed_idx):
        refs[int(labels[i])].append(pos[i])

    qset = set(queries)
    hit = n = 0
    dsum = 0.0
    app = wrong = 0
    captured = cap_right = cap_static_right = 0
    for c in order:
        if c in qset:
            true = int(labels[c])
            pred, grown = g.rank(c)
            hit += (pred == true); n += 1
            if grown:                                  # Q_A: growth answered this query
                captured += 1
                cap_right += int(pred == true)
                cap_static_right += int(g.rank_static(c) == true)
            own = refs.get(true)
            dsum += (min(abs(pos[c] - r) for r in own) / T) if own else 1.0
        tgt = None
        if mode in ("transport", "shuffled") and transport is not None and c in transport:
            tgt = int(transport[c]) if mode == "transport" else int(perm[int(transport[c])])
        elif mode == "top1" and c in qset:
            tgt = g.rank(c)[0]
        elif mode == "oracle" and c in qset:
            tgt = int(labels[c])
        if tgt is not None:
            ok = g.append(c, tgt)
            app += 1; wrong += int(not ok)
            if ok:
                refs[tgt].append(pos[c])               # only a correctly filed crop is a reference
    cr = captured / n if n else 0.0
    pe = cap_right / captured if captured else 0.0
    ap = cap_static_right / captured if captured else 0.0
    return {"R1_identity": hit / n if n else 0.0, "mean_distance": dsum / n if n else 0.0,
            "n_appended": app, "wrong_append": wrong / app if app else 0.0, "n_queries": n,
            "c": cr, "p_eff": pe, "a_plus": ap, "predicted_delta": cr * (pe - ap)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["transreid", "instructreid", "reid5o", "magiv3", "magiv2"])
    ap.add_argument("--link-backbone", default="magiv2", help="supplies the merge embedding")
    ap.add_argument("--link-source", choices=("magiv2", "self"), default="magiv2",
                    help="self: stage 2 merges on the evaluated backbone's own embedding")
    ap.add_argument("--stage1", choices=("magi", "self"), default="magi",
                    help="self: stage 1 clusters within a page on the evaluated backbone's features")
    ap.add_argument("--stage1-tau", type=float, default=0.7)
    ap.add_argument("--panels", type=Path, default=Path("results/panels_o2o"))
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--taus", nargs="+", type=float, default=None,
                    help="sweep the binder threshold; results nest under the value")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--strategies", nargs="+", default=["temporal", "random"])
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--series-file", type=Path, default=None,
                    help="explicit series list; Manga109's split file has no test key")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/transport.json"))
    args = ap.parse_args(argv)

    lspec = BACKBONE_REGISTRY[args.link_backbone]
    llm = load_checkpoint(str(Path(f"checkpoints/{args.link_backbone}/{args.config}/seed0/{args.checkpoint_name}")),
                          device=args.device)
    ltf = get_transforms(lspec.height, lspec.width, is_train=False, normalize_type=lspec.normalize)

    series = ([s.strip() for s in args.series_file.read_text().split("\n") if s.strip()]
              if args.series_file else load_split()["test"])
    link_cache: Dict[str, np.ndarray] = {}
    pc_cache: Dict[str, List] = {}
    for name in series:
        pf = args.panels / f"{name.replace(' ', '_')}.json"
        if not pf.exists():
            continue
        st = SeriesStream(args.data_root / name)
        link_cache[name] = np.asarray(FT.extract(llm.model, st, ltf, mode="none", batch_size=64,
                                                 device=args.device).bn, dtype=np.float64)
        pc_cache[name] = json.loads(pf.read_text())["magi_cluster_of_crop"]
    del llm

    results: Dict = {}
    for bb in args.backbones:
        spec = BACKBONE_REGISTRY[bb]
        lm = load_checkpoint(str(Path(f"checkpoints/{bb}/{args.config}/seed0/{args.checkpoint_name}")),
                             device=args.device)
        tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
        per: Dict = {}
        for name in link_cache:
            st = SeriesStream(args.data_root / name)
            labels, order = np.asarray(st.labels), st.reading_order
            pos = {c: t for t, c in enumerate(order)}
            feats = np.asarray(FT.extract(lm.model, st, tf, mode="none", batch_size=BATCH[bb],
                                          device=args.device).bn, dtype=np.float64)
            cell: Dict = {}
            for strat in args.strategies:
                for sd in args.seeds:
                    sm, q, go = split_seeds(labels, order, args.k, strat, sd)
                    seed_of = {i: int(c) for c, v in sm.items() for i in v}
                    seed_of.update({i: int(c) for c, v in go.items() for i in v})
                    lf = feats if args.link_source == "self" else link_cache[name]
                    for tau in (args.taus or [args.tau]):
                        # one threshold for both stages: they are cosine cuts on the same embedding
                        pc = (page_clusters_self(lf, st.crops, tau)
                              if args.stage1 == "self" else pc_cache[name])
                        tl = transport_labels(lf, pc, order, pos, seed_of, tau)
                        dst = cell.setdefault(strat, {})
                        if args.taus: dst = dst.setdefault("%.2f" % tau, {})
                        for mode in ("none", "transport", "shuffled", "top1", "oracle"):
                            dst.setdefault(mode, {})[str(sd)] = run(
                                feats, labels, order, args.k, strat, sd, mode=mode, transport=tl)
            per[name] = cell
            print(f"[transport] {bb} {name} done", flush=True)
        results[bb] = per
        del lm
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results))
    print(f"\n[transport] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
