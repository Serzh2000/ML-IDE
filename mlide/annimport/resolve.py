"""Поиск файла картинки по «ссылке» из файла разметки: путь, имя, имя без расширения, числовой номер."""
from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path

from .. import imgio


def norm_ref(ref) -> str:
    """Ссылка -> путь с «/», без кавычек, параметров URL и ./ в начале."""
    s = str(ref).strip().strip("'\"")
    if re.match(r"^[a-z][a-z0-9+.-]*://", s, re.I):
        s = urllib.parse.unquote(urllib.parse.urlparse(s).path)
    s = s.replace("\\", "/")
    s = re.sub(r"^(\./)+", "", s)
    return s


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


class ImageResolver:
    def __init__(self, roots=(), project=None):
        self.by_name: dict[str, list[str]] = {}
        self.by_stem: dict[str, list[str]] = {}
        self._seen: set[str] = set()
        for r in roots:
            self.add_root(r)
        if project is not None:
            for e in project.images:
                self._add(e.path)

    def _add(self, path: str):
        k = os.path.normcase(path)
        if k in self._seen:
            return
        self._seen.add(k)
        base = os.path.basename(path).lower()
        self.by_name.setdefault(base, []).append(path)
        self.by_stem.setdefault(_stem(base), []).append(path)

    def add_root(self, root):
        r = Path(root)
        if r.is_file() and imgio.is_image(r):
            self._add(str(r))
            return
        if not r.is_dir():
            return
        for dp, dn, fns in os.walk(r):
            dn[:] = [d for d in dn if not d.startswith(".")]
            for fn in fns:
                if imgio.is_image(fn):
                    self._add(os.path.join(dp, fn))

    @property
    def count(self) -> int:
        return len(self._seen)

    @staticmethod
    def _tail_score(ref_parts: list[str], path: str) -> int:
        pp = path.replace("\\", "/").lower().split("/")
        n = 0
        for i, (a, b) in enumerate(zip(reversed(ref_parts), reversed(pp))):
            if i == 0:                                       # имя файла: «a.rf.1b2c» в метке = «a.rf.1b2c.jpg» на диске
                same = a == b or a == _stem(b) or _stem(a) == _stem(b)
            else:
                same = a == b
            if not same:
                break
            n += 1
        return n

    def _pick(self, cands: list[str], ref_parts: list[str]):
        if len(cands) == 1:
            return cands[0], "ok"
        scores = [self._tail_score(ref_parts, c) for c in cands]
        best = max(scores)
        top = [c for c, s in zip(cands, scores) if s == best]
        return top[0], ("ok" if len(top) == 1 else "ambiguous")

    def find(self, ref, extra=()):
        """-> (путь или None, "ok" | "ambiguous" | "missing")."""
        for r in [ref, *extra]:
            if r is None or str(r).strip() == "":
                continue
            s = norm_ref(r)
            if os.path.isabs(s) or re.match(r"^[A-Za-z]:/", s):
                if os.path.isfile(s) and imgio.is_image(s):
                    return s, "ok"
            parts = [p for p in s.lower().split("/") if p and p != "."]
            if not parts:
                continue
            base = parts[-1]
            c = self.by_name.get(base)
            if c:
                return self._pick(c, parts)
            c = self.by_stem.get(base)                     # в метке нет расширения (в имени могут быть точки: a.rf.1b2c)
            if c:
                return self._pick(c, parts)
            st = _stem(base)
            c = self.by_stem.get(st)                       # другое расширение (img.png в метке, img.jpg на диске)
            if c:
                return self._pick(c, parts[:-1] + [st])
            if st.isdigit():                               # «12» ~ «000012.jpg»
                key = str(int(st))
                hits = [p for k, v in self.by_stem.items() if k.isdigit() and str(int(k)) == key for p in v]
                if hits:
                    return self._pick(hits, parts)
        return None, "missing"
