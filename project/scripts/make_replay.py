"""The two animated loops of the project page, computed rather than drawn.

Both loops replay one stream: eight crops of Bakuman chapter 1 (POPCharacters) in reading
order, against the galleries of the paper's protocols figure (Takagi, Mashiro and Azuki). The
stream was chosen from chapter 1 so that each protocol's characteristic behaviour shows up in
eight steps; every decision on it is computed here, none is drawn by hand:

- the features are MagiV2's released encoder with ViT-MAE masking off and the evaluation's
  identity BNNeck (unit norm), exactly what `scripts/evaluate.py --pretrained magiv2` uses;
- P1 and P2 rank a fixed gallery (P1 holds two crops of Takagi and Mashiro, P2 one seed each);
- P3 is `recognize.protocols.rules.fixed` at TAU_NOV on the stream alone;
- P4 starts from P2's seeds and files every query under its top-1 identity, after scoring it;
- Re:Cast keeps one L2-normalised mean per character and files a query only when MagiV2's page
  group (results/panels/Bakuman.json) links it to a crop already committed, as the `mustlink`
  policy of `run_p4` does.

The protocols themselves run on whole chapters; this is eight crops of one, which is what the
page's captions say.

    python project/scripts/make_replay.py     # from the repository root
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from memory_block.models.mae_masking import disable_mae_masking          # noqa: E402
from memory_block.models.model import MemoryConfig, MemoryEnhancedReID   # noqa: E402
from memory_block.training.dataset import get_transforms                 # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                        # noqa: E402
from recognize.data import SeriesStream                                  # noqa: E402
from recognize.features import identity_bnneck                           # noqa: E402
from recognize.protocol_constants import TAU_NOV                         # noqa: E402
from recognize.protocols.rules import fixed                              # noqa: E402

SERIES = ROOT / "Datasets" / "popcharacters" / "Bakuman"
PANELS = ROOT / "results" / "panels" / "Bakuman.json"
OUT_JSON = ROOT / "project" / "data" / "replay.json"
OUT_IMG = ROOT / "project" / "public" / "replay"
THUMB = (120, 160)                     # 3:4, twice the largest size the page draws a crop at

CHAR = {"Akito Takagi": "T", "Moritaka Mashiro": "M", "Miho Azuki": "A"}
PAGE = "Bakuman - c001 (web) - p%03d [Unknown]"

# (page, annotation index) of every crop the loops show. The gallery is the protocols figure's.
GALLERY = {"T1": (17, 0), "T2": (25, 4), "M1": (12, 1), "M2": (11, 3), "A1": (50, 2)}
P1_GALLERY = ["T1", "T2", "M1", "M2", "A1"]
SEEDS = ["T1", "M1", "A1"]              # P2, P4 and Re:Cast start from one seed per character
STREAM = {"q1": (8, 0), "q2": (17, 6), "q3": (25, 6), "q4": (29, 0),
          "q5": (43, 1), "q6": (44, 0), "q7": (47, 0), "q8": (50, 3)}


def crop_index(stream: SeriesStream) -> dict:
    """Crop id -> stream index, matched on page name and annotation index."""
    where = {(c.page_name, c.annotation_idx): i for i, c in enumerate(stream.crops)}
    return {cid: where[(PAGE % page, ann)] for cid, (page, ann) in {**GALLERY, **STREAM}.items()}


@torch.no_grad()
def features(stream: SeriesStream, idx: dict) -> dict:
    spec = BACKBONE_REGISTRY["magiv2"]
    cfg = MemoryConfig(backbone_type="magiv2", num_classes=1, feat_dim=spec.native_dim,
                       freeze_backbone=True, use_working_memory=False, use_episodic_memory=False,
                       image_height=spec.height, image_width=spec.width)
    model = identity_bnneck(MemoryEnhancedReID(cfg))
    disable_mae_masking(model)
    model.eval()
    tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
    ids = list(idx)
    batch = torch.stack([tf(stream.image(idx[c])) for c in ids])
    out = model(batch, use_memory=False)["bn_feat"].float()
    out = torch.nn.functional.normalize(out, dim=1).numpy()
    return {c: out[k].astype(np.float64) for k, c in enumerate(ids)}


def thumbnail(stream: SeriesStream, i: int, path: Path) -> None:
    """The annotated box with a little margin, cover-fitted to 3:4."""
    c = stream.crops[i]
    page = Image.open(c.image_path).convert("RGB")
    pw, ph = page.size
    cx, cy, bw, bh = c.bbox
    bw, bh = bw * 1.06, bh * 1.06
    box = (max(0, (cx - bw / 2) * pw), max(0, (cy - bh / 2) * ph),
           min(pw, (cx + bw / 2) * pw), min(ph, (cy + bh / 2) * ph))
    src = page.crop(tuple(int(v) for v in box))
    w, h = THUMB
    k = max(w / src.width, h / src.height)
    src = src.resize((max(1, round(src.width * k)), max(1, round(src.height * k))), Image.LANCZOS)
    left, top = (src.width - w) // 2, (src.height - h) // 2
    src.crop((left, top, left + w, top + h)).save(path, "WEBP", quality=82, method=6)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def top1(q: np.ndarray, entries: list, f: dict, filed: dict) -> dict:
    sims = [float(f[e] @ q) for e in entries]
    j = int(np.argmax(sims))
    return {"match": entries[j], "pred": filed[entries[j]], "sim": round(sims[j], 3)}


def main() -> int:
    stream = SeriesStream(SERIES)
    idx = crop_index(stream)
    truth = {c: CHAR[stream.crops[i].category] for c, i in idx.items()}
    groups = json.loads(PANELS.read_text())["magi_cluster_of_crop"]
    group = {c: groups[i] for c, i in idx.items()}
    f = features(stream, idx)
    queries = list(STREAM)

    # P3: the reference rule on the stream alone. The loop below repeats its centroid update only
    # to record the similarity each decision turned on; the assert keeps the two in step.
    clusters = fixed(np.stack([f[q] for q in queries]), tau=TAU_NOV).tolist()
    cents, counts, best = [], [], []
    for q in queries:
        s = [float(c @ f[q]) for c in cents]
        best.append(round(max(s), 3) if s else None)
        if s and max(s) >= TAU_NOV:
            j = int(np.argmax(s))
            cents[j] = cents[j] * counts[j] + f[q]
            cents[j] /= np.linalg.norm(cents[j]) + 1e-8
            counts[j] += 1
        else:
            j = len(cents)
            cents.append(f[q].copy())
            counts.append(1)
        assert j == clusters[len(best) - 1]

    steps = []
    p4_entries, p4_filed = list(SEEDS), {s: truth[s] for s in SEEDS}
    cast = {truth[s]: [s] for s in SEEDS}
    committed = {s: truth[s] for s in SEEDS}
    for k, q in enumerate(queries):
        step = {"q": q, "truth": truth[q],
                "p1": top1(f[q], P1_GALLERY, f, truth),
                "p2": top1(f[q], SEEDS, f, truth),
                "p3": {"cluster": clusters[k], "best": best[k], "open": clusters[k] not in clusters[:k]},
                "p4": top1(f[q], p4_entries, f, p4_filed)}
        p4_entries.append(q)
        p4_filed[q] = step["p4"]["pred"]

        chars = list(cast)
        means = {c: unit(np.mean([f[m] for m in cast[c]], axis=0)) for c in chars}
        sims = {c: round(float(means[c] @ f[q]), 3) for c in chars}
        pred = max(sims, key=sims.get)
        via = [m for m, c in committed.items() if group[m] is not None and group[m] == group[q]]
        commit = committed[via[0]] if via else None
        step["recast"] = {"pred": pred, "sims": sims, "commit": commit, "via": via[0] if via else None}
        if commit:
            cast[commit].append(q)
            committed[q] = commit
        steps.append(step)

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    crops = {}
    for c, i in idx.items():
        thumbnail(stream, i, OUT_IMG / f"{c}.webp")
        page, _ = {**GALLERY, **STREAM}[c]
        crops[c] = {"src": f"/replay/{c}.webp", "char": truth[c], "page": page}

    data = {
        "encoder": "MagiV2, released encoder",
        "series": "Bakuman", "chapter": 1, "tau": TAU_NOV,
        "names": {"T": "Takagi", "M": "Mashiro", "A": "Azuki"},
        "crops": crops, "stream": queries,
        "gallery": {"p1": P1_GALLERY, "seeds": SEEDS},
        "steps": steps,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1) + "\n")

    right = {p: sum(s[p]["pred"] == s["truth"] for s in steps) for p in ("p1", "p2", "p4", "recast")}
    wrong_adds = sum(s["p4"]["pred"] != s["truth"] for s in steps)
    print(f"[replay] Rank-1 over {len(steps)} queries: {right}; P4 wrong additions {wrong_adds}; "
          f"P3 clusters {len(set(clusters))}; Re:Cast commits "
          f"{[s['q'] for s in steps if s['recast']['commit']]}")
    print(f"[replay] wrote {OUT_JSON.relative_to(ROOT)} and {len(crops)} thumbnails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
