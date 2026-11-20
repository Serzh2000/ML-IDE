"""Контекст источника: что выбрал пользователь (файлы и папки) и удобные функции чтения для разборщиков форматов."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .. import imgio

# файлы, которые вообще могут содержать разметку; картинки и «мусор» пропускаем
ANN_EXTS = {".txt", ".xml", ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".yaml", ".yml", ".names", ".labels", ".list"}
SKIP_DIRS = {"__macosx", "node_modules", ".git", "__pycache__", "runs"}
MAX_FILES = 300_000


class SourceContext:
    def __init__(self, paths, images_dir=None):
        self.paths = [Path(p) for p in paths]
        self.images_dir = Path(images_dir) if images_dir else None
        self._files: list[Path] | None = None
        self._json: dict = {}
        self._text: dict = {}

    # ---- файлы
    def all_files(self) -> list[Path]:
        """Все файлы (не картинки) из выбранных путей, рекурсивно."""
        if self._files is None:
            out: list[Path] = []
            for p in self.paths:
                if p.is_file():
                    out.append(p)
                elif p.is_dir():
                    for dp, dn, fns in os.walk(p):
                        dn[:] = [d for d in dn if d.lower() not in SKIP_DIRS and not d.startswith(".")]
                        for fn in fns:
                            fp = Path(dp) / fn
                            if not imgio.is_image(fn):
                                out.append(fp)
                        if len(out) > MAX_FILES:
                            break
            self._files = sorted(out, key=lambda x: str(x).lower())
        return self._files

    def files(self, *exts: str) -> list[Path]:
        """Файлы с нужными расширениями (".json", ".csv"...); без аргументов - все возможные файлы разметки."""
        want = {e.lower() for e in exts} or ANN_EXTS
        return [f for f in self.all_files() if f.suffix.lower() in want]

    def image_roots(self) -> list[Path]:
        roots: list[Path] = []
        if self.images_dir:
            roots.append(self.images_dir)
        for p in self.paths:
            roots.append(p if p.is_dir() else p.parent)
        seen, out = set(), []
        for r in roots:
            k = os.path.normcase(str(r.resolve()))
            if k not in seen:
                seen.add(k)
                out.append(r)
        return out

    def parent_roots(self) -> list[Path]:
        """Родительские папки выбранного: картинки часто лежат рядом с папкой разметки (images/ и annotations/)."""
        out = []
        for r in self.image_roots():
            par = r.parent
            if par != r and len(par.parts) > 2:
                out.append(par)
        return out

    # ---- чтение
    def read_text(self, p, limit: int | None = None) -> str:
        key = (str(p), limit)
        if key in self._text:
            return self._text[key]
        raw = Path(p).read_bytes() if limit is None else open(p, "rb").read(limit)
        for enc in ("utf-8-sig", "utf-16", "cp1251", "latin-1"):
            try:
                txt = raw.decode(enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            txt = raw.decode("utf-8", "replace")
        if len(self._text) < 256:
            self._text[key] = txt
        return txt

    def read_json(self, p, max_bytes: int = 400_000_000):
        """JSON-файл -> объект или None (если не JSON или слишком большой). Результат кэшируется."""
        k = str(p)
        if k in self._json:
            return self._json[k]
        obj = None
        try:
            if Path(p).stat().st_size <= max_bytes:
                obj = json.loads(self.read_text(p))
        except (OSError, ValueError):
            obj = None
        if len(self._json) < 64:
            self._json[k] = obj
        return obj

    def read_jsonl(self, p, max_lines: int | None = None):
        """JSON Lines -> список объектов (битые строки пропускаются)."""
        out = []
        try:
            for i, ln in enumerate(self.read_text(p).splitlines()):
                if max_lines is not None and i >= max_lines:
                    break
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out
