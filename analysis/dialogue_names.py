"""Do spoken character names anchor identities across pages?

The appearance-based anchors are page-local. Must-link fires only where a gallery member shares the query's page group, which is why it reaches 9.5 % of the stream under random seeding and 2.3 % under chronological seeding, and why seed expansion rescues the first and not the second. Chronological seeding is the regime a reader actually encounters, so the question is whether any anchor works *across* pages.

Dialogue is the one candidate that is not an appearance signal at all. MagiV2 detects text boxes, reads them, and associates each to the character most likely to have spoken it; POPCharacters names its identities. A name in a bubble is therefore a label that arrives independently of what the character looks like, on whatever page the dialogue happens to fall.

The naive reading is almost certainly wrong, which is why this measures before anything is built on it. A name in a bubble is usually a *vocative* ("Takagi!" is spoken *to* Takagi, not *by* Takagi), so three rules are scored against ground truth rather than one:

    speaker     the character the bubble is associated with is the named one
    addressee   some other character on the page is the named one
    nearest     the character nearest the speaker, other than the speaker, is the named one

For each rule the script reports how often it fires and how often it is right, to set against the break-even append precision (40.9 % on MagiV2; the summary line gives the a+ bar of each cell). Nothing is built on a rule that does not clear it.

    python analysis/dialogue_names.py --device cuda --out results/names.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from recognize.backbones import BACKBONE_REGISTRY          # noqa: E402
from recognize.data import SeriesStream, load_split     # noqa: E402


def load_detector(device: str):
    from transformers import AutoModel
    spec = BACKBONE_REGISTRY["magiv2"]
    m = AutoModel.from_pretrained(spec.hf_repo, revision=spec.hf_revision, trust_remote_code=True)
    return m.to(device).eval()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower())


def name_tokens(mapping: Dict[str, str]) -> Dict[str, int]:
    """Surface form -> identity, for the surfaces a character could actually be called by.

    Two filters, both of which cost real precision when missing. A surface must be *capitalised in the annotation*, which keeps proper names and drops the descriptive scaffolding that annotation labels carry: "Ging and Mito's Grandmother" would otherwise contribute "and" as an identity surface, and "Moritaka's grandpa" would contribute "grandpa". And a shared surname identifies nobody: seven of Demon Slayer's eleven characters are a Kamado.
    """
    def parts(full: str):
        return [w for w in re.split(r"[^A-Za-z]+", full) if w]
    owners: Dict[str, set] = defaultdict(set)
    for cid, full in mapping.items():
        for w in parts(full):
            if w[:1].isupper() and len(norm(w).strip()) >= 3:
                owners[norm(w).strip()].add(int(cid))
    out: Dict[str, int] = {}
    for cid, full in mapping.items():
        whole = " ".join(norm(full).split())
        if whole:
            out[whole] = int(cid)
        for w in parts(full):
            t = norm(w).strip()
            if w[:1].isupper() and len(t) >= 3 and len(owners[t]) == 1:
                out[t] = int(cid)
    return out


def _within_one(a: str, b: str) -> bool:
    """True if `a` and `b` differ by at most one insertion, deletion or substitution."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        elif skipped:
            return False
        else:
            skipped = True; j += 1
    return True


def find_name(text: str, surfaces: Dict[str, int], fuzzy: bool) -> Optional[int]:
    """Identity named in `text`, by exact surface match and optionally by one OCR error.

    Names reach only about thirty percent of pages, and the mechanism that consumes them is bound by how many distinct stream positions carry a reference, not by precision: there is a margin of 46 to 59 % against a bar of 19.7 %. Tolerating a single character error is the cheapest way to buy pages. It is restricted to surfaces of five characters or more, because at three or four an edit of one is most of the token.
    """
    hit = next((cid for surf, cid in surfaces.items() if surf and f" {surf} " in text), None)
    if hit is not None or not fuzzy:
        return hit
    toks = [w for w in text.split() if len(w) >= 5]
    for surf, cid in surfaces.items():
        if len(surf) < 5 or " " in surf:
            continue
        for w in toks:
            if _within_one(w, surf):
                return cid
    return None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", nargs="+", default=None)
    # DIAG_CMD word-splits without quote processing, so series names with spaces need a file
    ap.add_argument("--series-file", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=Path("Datasets/popcharacters"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--fuzzy", action="store_true",
                    help="also match a surface at edit distance 1, for OCR errors")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/names.json"))
    args = ap.parse_args(argv)

    import torch
    from PIL import Image

    names = (args.series
             or ([n.strip() for n in args.series_file.read_text().split("\n") if n.strip()]
                 if args.series_file else load_split()["test"]))
    model = load_detector(args.device)
    summary = []
    for series in names:
        stream = SeriesStream(args.data_root / series)
        mapping = json.loads((args.data_root / series / "category_mapping.json").read_text())
        surfaces = name_tokens(mapping)
        labels = np.asarray(stream.labels)
        by_page: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(stream.crops):
            by_page[c.page_name].append(i)
        pages = sorted(by_page, key=lambda p: min(by_page[p]))

        stats = {r: {"fires": 0, "right": 0} for r in ("speaker", "addressee", "nearest")}
        speaker_calls: List[List[int]] = []
        n_named_bubbles = 0
        # per page: which identities a bubble names, and whether they are really on the page.
        # this is a presence signal, not an identification, so it seeds a candidate set.
        named_on_page: Dict[str, Dict[str, list]] = {}
        for start in range(0, len(pages), args.batch_size):
            chunk = pages[start:start + args.batch_size]
            imgs = [np.array(Image.open(stream.crops[by_page[p][0]].image_path).convert("L").convert("RGB"))
                    for p in chunk]
            with torch.no_grad():
                res = model.predict_detections_and_associations(imgs)
                texts = [r["texts"] for r in res]
                ocr = model.predict_ocr(imgs, texts)
            for p, r, page_ocr in zip(chunk, res, ocr):
                idxs = by_page[p]
                w, h = Image.open(stream.crops[idxs[0]].image_path).size
                gt = []
                for i in idxs:
                    cx, cy, bw, bh = stream.crops[i].bbox
                    gt.append(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
                det = [[float(v) for v in b] for b in r.get("characters", [])]
                # detection -> annotated crop
                det2crop: Dict[int, int] = {}
                for di, db in enumerate(det):
                    best, bi = 0.0, None
                    for k, gb in zip(idxs, gt):
                        v = iou(db, gb)
                        if v > best: best, bi = v, k
                    if bi is not None and best >= args.min_iou:
                        det2crop[di] = bi
                assoc = {int(t): int(c) for t, c in r.get("text_character_associations", [])}
                for ti, line in enumerate(page_ocr):
                    text = " " + " ".join(norm(line).split()) + " "
                    hit = find_name(text, surfaces, args.fuzzy)
                    if hit is None:
                        continue
                    n_named_bubbles += 1
                    spk = det2crop.get(assoc.get(ti, -1))
                    if spk is not None:
                        stats["speaker"]["fires"] += 1
                        stats["speaker"]["right"] += int(labels[spk] == hit)
                        # (crop, identity the bubble names, whether that is the crop's true identity).
                        # The aggregate counts cannot drive an append; this can.
                        speaker_calls.append([int(spk), int(hit), int(labels[spk] == hit)])
                        others = [c for c in det2crop.values() if c != spk]
                        if others:
                            stats["addressee"]["fires"] += 1
                            stats["addressee"]["right"] += int(any(labels[o] == hit for o in others))
                            sb = gt[idxs.index(spk)]
                            near = min(others, key=lambda o: abs(gt[idxs.index(o)][0]-sb[0]))
                            stats["nearest"]["fires"] += 1
                            stats["nearest"]["right"] += int(labels[near] == hit)
                            rec = named_on_page.setdefault(p, {"named": [], "present": []})
                            rec["named"].append(int(hit))
                            rec["present"].append(int(any(labels[o] == hit for o in others)))
        summary.append((series, n_named_bubbles, stats, named_on_page,
                        {p: [int(i) for i in v] for p, v in by_page.items()},
                        [int(x) for x in labels], speaker_calls))
        print(f"[names] {series}: {n_named_bubbles} bubbles containing a character name; "
              + ", ".join(f"{r} {s['right']}/{s['fires']}" for r, s in stats.items())
              + f"; {len(named_on_page)} pages carry a name", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        [{"series": s, "named_bubbles": n, "rules": st, "named_on_page": nop,
          "crops_by_page": cbp, "labels": lab, "speaker_calls": sc}
         for s, n, st, nop, cbp, lab, sc in summary], indent=1))
    print(f"\n{'rule':<12}{'fires':>8}{'correct':>10}   (break-even is a+ in the deployment cell:"
          f" 19.7-36.7 % under Seq-T, 23.1-43.7 % under Seq-R)")
    for rule in ("speaker", "addressee", "nearest"):
        f = sum(s[rule]["fires"] for _, _, s, *_ in summary)
        r = sum(s[rule]["right"] for _, _, s, *_ in summary)
        print(f"{rule:<12}{f:>8}{100*r/max(1,f):>9.1f}%")
    print(f"\ntotal bubbles containing a character name: {sum(n for _, n, *_ in summary)}")
    print(f"[names] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
