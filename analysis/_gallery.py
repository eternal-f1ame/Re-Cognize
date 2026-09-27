"""Gallery semantics for the analysis scripts, written once, with the two labels kept apart.

Every entry in a grown gallery carries two identities and they are not the same thing.

    filed   the identity the entry is stored under, which is what a match returns
    true    the identity the crop actually is, which is what scoring compares against

For a seed the two coincide. For an appended crop they coincide only when the append was correct. Reading the wrong one produces one of two defects:

  * `p_eff` read from the true identity instead of the filed one. An exemplar that genuinely is the query's character but sits under another name still returns the wrong answer, so counting it as a success inflates capture precision (from about 0.25 to 0.55).
  * a reference counted for the identity it is filed under. Append-by-top-1 files 467 crops, 78.8 % of them wrongly; crediting each as a reference for whatever name it lands under makes the worst policy appear to remove the most distance.

Both are avoided by never reading `g_id` or `labels` at a call site. `Gallery` owns the pair, and the three quantities that depend on getting it right are computed here.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np


class Gallery:
    """A P4 gallery: protected seeds, FIFO growth per identity, ranking against a snapshot."""

    def __init__(self, feats: np.ndarray, labels, seed_idx, b_max: int = 50):
        self.feats = feats
        self.labels = np.asarray(labels)
        self.b_max = b_max
        cap = len(feats) * 2 + len(seed_idx) + 4
        self._f = np.zeros((cap, feats.shape[1]))
        self._filed = np.zeros(cap, dtype=np.int64)
        self._true = np.zeros(cap, dtype=np.int64)
        self._seed = np.zeros(cap, dtype=bool)
        self._on = np.zeros(cap, dtype=bool)
        self._n = 0
        self._buf: Dict[int, deque] = defaultdict(deque)
        for i in sorted(seed_idx):
            self._f[self._n] = feats[i]
            self._filed[self._n] = self._true[self._n] = self.labels[i]
            self._seed[self._n] = self._on[self._n] = True
            self._n += 1
        self.n_seed_rows = self._n
        self._static = np.arange(self.n_seed_rows)

    def rank(self, q: int):
        """(filed identity of the winner, whether the winner is a grown entry)."""
        idx = np.flatnonzero(self._on[: self._n])
        w = idx[int(np.argmax(self._f[idx] @ self.feats[q]))]
        return int(self._filed[w]), not bool(self._seed[w])

    def rank_static(self, q: int) -> int:
        """What the frozen seed gallery alone would answer. The a+ term needs this."""
        s = self._static
        return int(self._filed[s[int(np.argmax(self._f[s] @ self.feats[q]))]])

    def append(self, crop: int, filed_as: int) -> bool:
        """File `crop` under `filed_as`. Returns whether the append was correct."""
        buf = self._buf[filed_as]
        if len(buf) >= self.b_max:
            self._on[buf.popleft()] = False
        self._f[self._n] = self.feats[crop]
        self._filed[self._n] = filed_as
        self._true[self._n] = self.labels[crop]
        self._on[self._n] = True
        buf.append(self._n)
        self._n += 1
        return int(self.labels[crop]) == int(filed_as)

    def references_of(self, identity: int) -> List[int]:
        """Rows that are genuinely `identity` AND filed under it: the only ones that answer for it.

        This is the definition the distance law of `seed_distance.py` is measured under, where every
        reference is a seed and both conditions hold by construction.
        """
        rows = np.flatnonzero(self._on[: self._n])
        ok = rows[(self._filed[rows] == identity) & (self._true[rows] == identity)]
        return [int(r) for r in ok]


def self_test() -> None:
    """Three identities, hand-checkable. Run with `python analysis/_gallery.py`."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=(6, 4))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    labels = [0, 1, 2, 0, 1, 2]
    g = Gallery(f, labels, seed_idx=[0, 1, 2], b_max=50)

    assert g.n_seed_rows == 3
    assert [len(g.references_of(i)) for i in (0, 1, 2)] == [1, 1, 1], "seeds are references"

    assert g.append(3, 0) is True, "crop 3 is identity 0, filed as 0 -> correct"
    assert len(g.references_of(0)) == 2, "a correct append adds a reference"

    assert g.append(4, 2) is False, "crop 4 is identity 1, filed as 2 -> wrong"
    assert len(g.references_of(2)) == 1, "a misfiled crop is NOT a reference for its filed identity"
    assert len(g.references_of(1)) == 1, "nor for its true identity, since it answers as 2"

    pred, grown = g.rank(5)
    assert isinstance(pred, int) and isinstance(grown, bool)
    assert g.rank_static(5) in (0, 1, 2), "static ranking ignores grown rows"

    # b_max eviction keeps the seed and drops the oldest growth
    h = Gallery(f, labels, seed_idx=[0], b_max=1)
    h.append(3, 0); h.append(1, 0)
    assert len(h.references_of(0)) == 1, "evicting the oldest growth leaves the seed plus one"
    print("_gallery self-test: ok")


if __name__ == "__main__":
    self_test()
