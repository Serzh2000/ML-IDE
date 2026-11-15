"""Near-duplicate detection and leak-free grouping for dataset splits (pure logic, no Qt)."""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import imgio


# ---------------------------------------------------------------- hashing
def dhash(img_bgr, size: int = 8) -> int:
    """size*size-bit difference hash (default 64 bit) of a BGR or gray image."""
    g = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hash_file(path) -> int | None:
    """dhash of a file using a cheap reduced-size grayscale decode (JPEG: DCT scaling)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        g = cv2.imdecode(data, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if g is None:
            g = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
        return dhash(g)
    except Exception:
        return None


def hash_files(paths, progress=None, workers: int = 8):
    """list[int|None] aligned with paths. progress(done,total)->False cancels (returns None)."""
    paths = list(paths)
    out: list = [None] * len(paths)
    total = len(paths)
    chunk = 256
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, total, chunk):
            part = paths[i:i + chunk]
            for j, h in enumerate(ex.map(_hash_file, part)):
                out[i + j] = h
            if progress is not None and progress(min(i + chunk, total), total) is False:
                return None
    return out


# ---------------------------------------------------------------- matching
if hasattr(np, "bitwise_count"):
    def _popcount(x):
        return np.bitwise_count(x)
else:  # numpy < 2.0
    _TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(x):
        return _TABLE[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1, dtype=np.uint8)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def cluster_hashes(hashes, threshold: int = 6, block: int = 256, progress=None):
    """Connected-component labels for uint64 hashes (pairs with hamming <= threshold are linked).

    Returns an int array `labels` (label = smallest member index of the component) or None
    if cancelled. Exact duplicates are collapsed first; the rest is a blocked, vectorised
    all-pairs popcount (n^2/2 uint64 ops in numpy, no Python pair loop).
    """
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    uniq, first, inv = np.unique(h, return_index=True, return_inverse=True)
    m = len(uniq)
    uf = _UF(m)
    for i0 in range(0, m, block):
        i1 = min(i0 + block, m)
        x = uniq[i0:i1, None] ^ uniq[None, i0:]
        d = _popcount(x)
        ii, jj = np.nonzero(d <= threshold)
        jj = jj + i0
        ii = ii + i0
        sel = jj > ii
        for a, b in zip(ii[sel].tolist(), jj[sel].tolist()):
            uf.union(a, b)
        if progress is not None and progress(i1, m) is False:
            return None
    comp_u = np.array([uf.find(i) for i in range(m)], dtype=np.int64)
    # component label per element = smallest original index in the component
    comp = comp_u[inv]
    minidx = np.full(m, n, dtype=np.int64)
    np.minimum.at(minidx, comp, np.arange(n))
    return minidx[comp]


def find_near_duplicates(paths, threshold: int = 6, progress=None) -> list[list[str]]:
    """Groups (size >= 2) of near-duplicate images. Unreadable files are ignored.

    progress(done, total) -> False cancels (returns []). Hashing is the first half of the
    reported total, matching the second.
    """
    paths = [str(p) for p in paths]
    n = len(paths)
    cancelled = False

    def p1(d, t):
        nonlocal cancelled
        if progress is not None and progress(d, 2 * n) is False:
            cancelled = True
            return False
        return True

    hs = hash_files(paths, p1)
    if hs is None or cancelled:
        return []
    ok = [i for i, h in enumerate(hs) if h is not None]
    if len(ok) < 2:
        return []
    arr = np.array([hs[i] for i in ok], dtype=np.uint64)

    def p2(d, t):
        nonlocal cancelled
        if progress is not None and progress(n + int(n * d / max(1, t)), 2 * n) is False:
            cancelled = True
            return False
        return True

    labels = cluster_hashes(arr, threshold, progress=p2)
    if labels is None or cancelled:
        return []
    groups: dict[int, list[str]] = {}
    for k, lab in enumerate(labels.tolist()):
        groups.setdefault(lab, []).append(paths[ok[k]])
    return sorted((g for g in groups.values() if len(g) >= 2), key=lambda g: g[0])


# ---------------------------------------------------------------- names
_TRAIL = re.compile(r"[\s_\-.]*[fF]?\d+$")


def source_key(path) -> str:
    """Group key for frames of the same video / sequence: '<parent folder>/<stem without trailing index>'.

    The trailing index is an optional separator run (space _ - .), an optional 'f', then
    digits at the very end of the stem; everything lower-cased. Examples (same folder 'v'):
      clip_000123.jpg, clip-123.png -> 'v/clip'      frame_000045 -> 'v/frame'
      video1_f0003 -> 'v/video1'   (video2_f0003 -> 'v/video2' stays separate)
      IMG_0001.jpg -> 'v/img'      (all IMG_#### in one folder form one group; turn
                                    use_names off if that is a flat photo dump)
      000123.jpg -> 'v/'           (pure number: grouped by folder only)
      cat.jpg -> 'v/cat'           (no index: its own key)
    """
    p = Path(path)
    stem = p.stem
    base = _TRAIL.sub("", stem) if re.search(r"\d$", stem) else stem
    return f"{p.parent.name.lower()}/{base.lower()}"


def build_groups(paths, use_hash: bool = True, hash_threshold: int = 6,
                 use_names: bool = True, progress=None) -> dict[str, str]:
    """Map path -> group id ('g00000' = index of the first member). Merges near-duplicate
    clusters and filename-sequence groups with union-find; unrelated images are singletons."""
    paths = [str(p) for p in paths]
    n = len(paths)
    uf = _UF(n)
    if use_names:
        seen: dict[str, int] = {}
        for i, p in enumerate(paths):
            k = source_key(p)
            if k in seen:
                uf.union(seen[k], i)
            else:
                seen[k] = i
    if use_hash and n > 1:
        idx = {p: i for i, p in enumerate(paths)}
        for grp in find_near_duplicates(paths, hash_threshold, progress):
            a = idx[grp[0]]
            for q in grp[1:]:
                uf.union(a, idx[q])
    return {p: f"g{uf.find(i):05d}" for i, p in enumerate(paths)}


# ---------------------------------------------------------------- splitting
def split_groups(items, group_of, fractions: dict, seed: int = 0) -> dict[str, list]:
    """Split `items` into named splits keeping every group whole.

    >>> items = [("a", 1), ("b", 1), ("c", 2), ("d", 2), ("e", 3)]
    >>> r = split_groups(items, lambda it: it[1], {"train": 0.6, "val": 0.4}, seed=1)
    >>> sorted(len(v) for v in r.values())
    [2, 3]

    Groups are placed largest first (equal sizes in seeded random order) into the split with
    the biggest remaining deficit vs. its target share. Deterministic per seed; splits that got
    nothing are omitted; items keep their input order inside each split.
    """
    items = list(items)
    names = [k for k, v in fractions.items() if v > 0]
    if not items or not names:
        return {}
    tot = float(sum(fractions[k] for k in names))
    n = len(items)
    target = {k: n * fractions[k] / tot for k in names}
    groups: dict = {}
    for i, it in enumerate(items):
        groups.setdefault(group_of(it), []).append(i)
    glist = list(groups.values())
    random.Random(seed).shuffle(glist)
    glist.sort(key=len, reverse=True)  # stable: ties keep shuffled order
    got = {k: 0 for k in names}
    assign: dict[str, list[int]] = {k: [] for k in names}
    for g in glist:
        best = max(names, key=lambda k: (target[k] - got[k]) / target[k])
        assign[best].extend(g)
        got[best] += len(g)
    return {k: [items[i] for i in sorted(v)] for k, v in assign.items() if v}
