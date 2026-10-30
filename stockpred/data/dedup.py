"""Near-duplicate clustering of syndicated stories (MinHash + LSH banding + union-find).

Each cluster is an "event"; counting events instead of articles stops one wire story
republished by five outlets from inflating buzz.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

import numpy as np

_PRIME = (1 << 61) - 1
_WORD = re.compile(r"[a-z0-9']+")


def shingles(text: str, k: int = 3) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") % _PRIME


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64).astype(object)
        self.b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64).astype(object)

    def signature(self, sh: set[str]) -> tuple[int, ...]:
        if not sh:
            return tuple([0] * self.num_perm)
        hs = [_h64(s) for s in sh]
        return tuple(min((a * h + b) % _PRIME for h in hs) for a, b in zip(self.a, self.b))


def jaccard_est(s1: tuple[int, ...], s2: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(s1, s2)) / len(s1)


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: lexicographically smaller id
            lo, hi = sorted((ra, rb))
            self.p[hi] = lo


def cluster_events(
    items: Iterable[tuple[str, str]], threshold: float = 0.6, num_perm: int = 64, bands: int = 16
) -> dict[str, str]:
    """items: (article_id, text). Returns {article_id: event_id} (event_id = smallest id in cluster)."""
    mh = MinHasher(num_perm)
    rows = max(1, num_perm // bands)
    sigs: dict[str, tuple[int, ...]] = {}
    buckets: dict[tuple, list[str]] = defaultdict(list)
    uf = _UF()
    for aid, text in items:
        uf.find(aid)
        sig = mh.signature(shingles(text))
        sigs[aid] = sig
        if not any(sig):
            continue
        for bi in range(bands):
            key = (bi, sig[bi * rows : (bi + 1) * rows])
            buckets[key].append(aid)
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        first = ids[0]
        for other in ids[1:]:
            if jaccard_est(sigs[first], sigs[other]) >= threshold:
                uf.union(first, other)
    return {aid: uf.find(aid) for aid in sigs}
