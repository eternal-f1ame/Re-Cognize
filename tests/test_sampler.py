"""PKSampler without replacement."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest


def _dataset(counts):
    idx, char_to_samples = 0, {}
    for lab, n in counts.items():
        char_to_samples[lab] = list(range(idx, idx + n)); idx += n
    return SimpleNamespace(char_to_samples=char_to_samples)


class TestPKSampler:
    def test_excludes_small_identities_and_yields_distinct_contiguous_windows(self):
        from memory_block.training.dataset import PKSampler
        ds = _dataset({0: 2, 1: 3, 2: 4, 3: 10, 4: 7, 5: 4})
        s = PKSampler(ds, p=2, k=4)
        assert set(s.labels) == {2, 3, 4, 5} and s.n_excluded == 2 and s.p == 2
        random.seed(0)
        owner = {i: lab for lab, idxs in ds.char_to_samples.items() for i in idxs}
        for _ in range(500):
            batch = list(iter(s))
            assert len(batch) == len(s) == (4 // 2) * 2 * 4
            for g in range(0, len(batch), 4):
                group = batch[g:g + 4]
                labs = {owner[i] for i in group}
                assert len(labs) == 1 and len(set(group)) == 4              # one identity, distinct crops
                assert group == list(range(group[0], group[0] + 4))          # contiguous window (tracklet)
                assert group[0] >= ds.char_to_samples[labs.pop()][0]

    def test_ragged_tail_is_dropped(self):
        from memory_block.training.dataset import PKSampler
        s = PKSampler(_dataset({0: 5, 1: 5, 2: 5}), p=2, k=4)
        assert len(s) == 8 and len(list(iter(s))) == 8

    def test_no_identity_is_empty(self):
        from memory_block.training.dataset import PKSampler
        s = PKSampler(_dataset({0: 2, 1: 3}), p=2, k=4)
        assert s.labels == [] and len(s) == 0 and list(iter(s)) == []
