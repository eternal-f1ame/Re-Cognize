"""Merge page clusters across pages, and see whether averaging beats linking crop to crop.

Stage 1 groups crops inside a page at 90.7 % must-link precision, but a within-page link cannot help: stream distance is a minimum over an identity's references, so a link that does not cross a page moves nothing. Page-group propagation shows it: it triples the appends and moves mean distance only from 0.0869 to 0.0882.

Flat cross-page clustering does cross pages, and it is the baseline to beat. Running MagiV2's own clustering over chunks of pages and keeping its labels unprefixed gives 43.0 % must-link at a median span of 0.024 of the stream, 38.9 % at 0.073, and 34.2 % at 0.187. All three clear the 19.7 to 25.6 % break-even of the weak backbones' galleries.

Two-stage should beat that, for a reason `prototype_gallery.py` measures. The cast sheet, one L2-normalised mean per identity instead of a bag of exemplars, is worth +6.81 identity Rank-1 on MagiV2 and is positive on five of five backbones, because a mean estimates an identity better than any single crop does. A page cluster is a small cast sheet. Merging cluster means therefore decides each link on two or three crops of evidence rather than one.

The prediction is specific, and so is its limit: half of all page clusters hold exactly one crop, so the averaging advantage applies to about half the merge decisions and to none of the rest. Singleton-seeded and multi-crop merges are reported separately for that reason, because a single pooled number would hide the mechanism.

Merges are constrained to mutual nearest neighbours above a threshold. Agglomerative merging is transitive, and one bad merge unions two identities for every query afterwards: the same defect that costs the cannot-link half of joint assignment 15 to 34 points.

    python analysis/stage2_merge.py --device cuda --panels results/panels_o2o --out results/stage2.json
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
from recognize.protocols import split_seeds                       # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402

TAUS = (0.50, 0.60, 0.70, 0.80, 0.90)


def merge_mnn(means: np.ndarray, pages: List[str], tau: float, rounds: int = 8):
    """Agglomerate cluster means by mutual nearest neighbour above `tau`. Returns a parent map."""
    n = len(means)
    parent = list(range(n))
    cur = means.copy()
    alive = np.ones(n, dtype=bool)
    page_of = [{p} for p in pages]
    size = np.ones(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for _ in range(rounds):
        idx = np.flatnonzero(alive)
        if len(idx) < 2:
            break
        S = cur[idx] @ cur[idx].T
        np.fill_diagonal(S, -1.0)
        for a in range(len(idx)):                       # a merge must not join two same-page clusters
            for b in range(len(idx)):
                if a != b and page_of[idx[a]] & page_of[idx[b]]:
                    S[a, b] = -1.0
        nn = S.argmax(axis=1)
        merged = False
        for a in range(len(idx)):
            b = int(nn[a])
            if b <= a and int(nn[b]) == a:
                continue
            if int(nn[b]) != a or S[a, b] < tau:
                continue
            ia, ib = find(int(idx[a])), find(int(idx[b]))
            if ia == ib:
                continue
            w = size[ia] + size[ib]
            v = (cur[ia] * size[ia] + cur[ib] * size[ib]) / w
            v /= max(np.linalg.norm(v), 1e-12)
            parent[ib] = ia; cur[ia] = v; size[ia] = w
            page_of[ia] |= page_of[ib]; alive[ib] = False
            merged = True
        if not merged:
            break
    return [find(i) for i in range(n)]


def score(groups: Dict[int, List[int]], members: List[List[int]], labels, pos, T, sizes):
    """Must-link precision of the merged clusters, split by whether a singleton is involved."""
    within = [0, 0]; cross = [0, 0]; spans = []
    single = [0, 0]; multi = [0, 0]
    for g, cids in groups.items():
        crops = [(c, ci) for ci in cids for c in members[ci]]
        for a in range(len(crops)):
            for b in range(a + 1, len(crops)):
                (i, ca), (j, cb) = crops[a], crops[b]
                same = int(labels[i] == labels[j])
                if ca == cb:
                    within[0] += same; within[1] += 1
                else:
                    cross[0] += same; cross[1] += 1
                    spans.append(abs(pos[i] - pos[j]) / T)
                    if sizes[ca] == 1 or sizes[cb] == 1:
                        single[0] += same; single[1] += 1
                    else:
                        multi[0] += same; multi[1] += 1
    f = lambda x: (x[0] / x[1] if x[1] else None, x[1])
    return {"within_ml": f(within)[0], "within_pairs": within[1],
            "cross_ml": f(cross)[0], "cross_pairs": cross[1],
            "cross_ml_singleton": f(single)[0], "singleton_pairs": single[1],
            "cross_ml_multi": f(multi)[0], "multi_pairs": multi[1],
            "span_p50": float(np.median(spans)) if spans else 0.0,
            "span_mean": float(np.mean(spans)) if spans else 0.0,
            "n_groups": len(groups)}



def reach(groups: Dict[int, List[int]], members: List[List[int]], labels, order, pos, T, k, seed):
    """Can a merged group reach a seed, and is the identity it transports the right one?

    A merge only moves anything if one of its members is labelled. Under chronological seeding every
    seed sits in the opening pages, so this is the question that decides whether the relation is
    usable at all: precision on the relation is worth nothing if the groups never touch a seed.

    Reports, per gallery seed: the share of groups holding at least one seed, the share of unlabelled
    crops those groups reach, the precision of the label such a group would transport, and the
    distance that label would travel. The last is the quantity page-group propagation does not move.
    """
    seed_map, queries, gallery_only = split_seeds(labels, order, k, "temporal", seed)
    seed_of = {}
    for c, idx in seed_map.items():
        for i in idx:
            seed_of[i] = int(c)
    for c, idx in gallery_only.items():
        for i in idx:
            seed_of[i] = int(c)
    qset = set(queries)

    g_with_seed = 0
    reached = right = 0
    spans = []
    for g, cids in groups.items():
        crops = [c for ci in cids for c in members[ci]]
        seeds = [(c, seed_of[c]) for c in crops if c in seed_of]
        if not seeds:
            continue
        g_with_seed += 1
        # the group transports its majority seed identity
        vote: Dict[int, int] = {}
        for _, lab in seeds:
            vote[lab] = vote.get(lab, 0) + 1
        tgt = max(vote, key=vote.get)
        anchor = min((pos[c] for c, _ in seeds), key=lambda x: x)
        for c in crops:
            if c in seed_of or c not in qset:
                continue
            reached += 1
            right += int(labels[c] == tgt)
            spans.append(abs(pos[c] - anchor) / T)
    return {"groups_with_seed": g_with_seed, "n_groups": len(groups),
            "group_share": g_with_seed / max(len(groups), 1),
            "reached": reached, "n_queries": len(qset),
            "reach_share": reached / max(len(qset), 1),
            "transport_precision": right / reached if reached else None,
            "transport_span_p50": float(np.median(spans)) if spans else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", type=Path, default=Path("results/panels_o2o"))
    ap.add_argument("--backbone", default="magiv2", help="supplies the crop embedding used to merge")
    ap.add_argument("--config", default="finetuned")
    ap.add_argument("--checkpoint-name", default="epoch_0200.pth")
    ap.add_argument("--taus", nargs="+", type=float, default=list(TAUS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/stage2.json"))
    args = ap.parse_args(argv)

    spec = BACKBONE_REGISTRY[args.backbone]
    lm = load_checkpoint(str(Path(f"checkpoints/{args.backbone}/{args.config}/seed0/{args.checkpoint_name}")),
                         device=args.device)
    tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)

    out: Dict = {}
    for name in load_split()["test"]:
        pf = args.panels / f"{name.replace(' ', '_')}.json"
        if not pf.exists():
            print(f"[stage2] {pf} missing, skipping", flush=True); continue
        cl = json.loads(pf.read_text())["magi_cluster_of_crop"]
        stream = SeriesStream(args.data_root / name)
        labels = np.asarray(stream.labels)
        order = stream.reading_order
        pos = {c: t for t, c in enumerate(order)}
        T = max(1, len(order) - 1)
        feats = np.asarray(FT.extract(lm.model, stream, tf, mode="none",
                                      batch_size=64, device=args.device).bn, dtype=np.float64)

        by_cluster: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(cl):
            if c:
                by_cluster[c].append(i)
        keys = sorted(by_cluster)
        members = [by_cluster[k] for k in keys]
        sizes = [len(m) for m in members]
        pages = [k.split("#")[0] for k in keys]
        means = np.stack([feats[m].mean(0) for m in members])
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)

        cell: Dict = {}
        for tau in args.taus:
            par = merge_mnn(means, pages, tau)
            groups: Dict[int, List[int]] = defaultdict(list)
            for ci, root in enumerate(par):
                groups[root].append(ci)
            cell[f"{tau:.2f}"] = score(groups, members, labels, pos, T, sizes)
            cell[f"{tau:.2f}"]["reach"] = {
                str(sd): reach(groups, members, labels, order, pos, T, args.k, sd)
                for sd in range(args.seeds)}
            r = cell[f"{tau:.2f}"]
            print(f"[stage2] {name} tau={tau:.2f} cross {100*(r['cross_ml'] or 0):5.1f}% "
                  f"({r['cross_pairs']:5d})  multi {100*(r['cross_ml_multi'] or 0):5.1f}%  "
                  f"single {100*(r['cross_ml_singleton'] or 0):5.1f}%  span {r['span_p50']:.3f}", flush=True)
        out[name] = cell
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\n[stage2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
