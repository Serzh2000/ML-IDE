"""Data model: annotations, images, classes, project (JSON on disk)."""
from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import imgio, textio

PALETTE = [
    "#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93", "#ff924c",
    "#52b788", "#f15bb5", "#00bbf9", "#9b5de5", "#fee440", "#00f5d4",
]


@dataclass
class Annotation:
    cls: int
    kind: str                      # "box" -> pts=[[x1,y1],[x2,y2]];  "poly" -> pts=[[x,y],...];
                                   # "obb" -> 4 угла повёрнутого прямоугольника по кругу;  "pose" -> рамка (как box) + kps
    pts: list
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    suggested: bool = False        # produced by auto-annotation, awaiting review
    conf: float | None = None
    kps: list | None = None        # pose: [[x, y, v], ...]  v: 0 нет, 1 закрыта, 2 видна

    def bbox(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return min(xs), min(ys), max(xs), max(ys)

    def polygon(self):
        if self.kind in ("box", "pose"):
            x1, y1, x2, y2 = self.bbox()
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return [list(p) for p in self.pts]

    def to_dict(self):
        d = {"cls": self.cls, "kind": self.kind,
             "pts": [[round(float(x), 2), round(float(y), 2)] for x, y in self.pts]}
        if self.suggested:
            d["suggested"] = True
        if self.conf is not None:
            d["conf"] = round(float(self.conf), 4)
        if self.kps is not None:
            d["kps"] = [[round(float(x), 2), round(float(y), 2), int(v)] for x, y, v in self.kps]
        return d

    @staticmethod
    def from_dict(d):
        return Annotation(cls=int(d["cls"]), kind=d["kind"], pts=[list(p) for p in d["pts"]],
                          suggested=bool(d.get("suggested", False)), conf=d.get("conf"),
                          kps=[list(k) for k in d["kps"]] if d.get("kps") is not None else None)


@dataclass
class ImageEntry:
    path: str                      # absolute path in memory
    w: int
    h: int
    anns: list = field(default_factory=list)
    reviewed: bool = False
    tag: int | None = None         # класс изображения целиком (задача классификации)

    @property
    def name(self):
        return Path(self.path).name

    def snapshot(self):
        return copy.deepcopy({"anns": [a.to_dict() for a in self.anns], "tag": self.tag})

    def restore(self, snap):
        if isinstance(snap, dict):
            self.anns = [Annotation.from_dict(d) for d in snap["anns"]]
            self.tag = snap.get("tag")
        else:                                   # старый формат: только список объектов
            self.anns = [Annotation.from_dict(d) for d in snap]


class Project:
    def __init__(self, name="Без названия"):
        self.name = name
        self.path: Path | None = None
        self.classes: list[dict] = []
        self.images: list[ImageEntry] = []
        self.presets: dict[str, dict] = {}      # user augmentation presets
        self.export: dict = {}                  # last export settings
        self.keypoints: list[str] = []          # имена ключевых точек (задача pose)
        self.skeleton: list[list[int]] = []     # пары индексов точек, соединяемых линиями
        self.flip_pairs: list[list[int]] = []   # пары «левая-правая» точки для отражения
        self.dirty = False
        self.recovered_from_backup = False

    # ---- потерянные файлы
    def missing_images(self):
        return [e for e in self.images if not Path(e.path).exists()]

    def relink(self, folder) -> tuple[int, int]:
        """Ищет пропавшие изображения по имени файла в папке (и подпапках). Возвращает (найдено, осталось)."""
        miss = self.missing_images()
        if not miss:
            return 0, 0
        index: dict[str, list[str]] = {}
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if imgio.is_image(f):
                    index.setdefault(f.lower(), []).append(os.path.join(root, f))
        fixed = 0
        for e in miss:
            cands = index.get(Path(e.path).name.lower()) or []
            want = Path(e.path).parts[::-1]

            def score(c):
                n = 0
                for a, b in zip(want, Path(c).parts[::-1]):
                    if a.lower() != b.lower():
                        break
                    n += 1
                return n

            ok = []
            for c in sorted(cands, key=score, reverse=True):
                try:
                    if imgio.size_of(c) == (e.w, e.h):             # другой размер - это другая картинка с тем же именем
                        ok.append(c)
                except Exception:  # noqa
                    continue
            if ok:
                e.path = str(Path(ok[0]).resolve())
                fixed += 1
        if fixed:
            self.dirty = True
        return fixed, len(miss) - fixed

    # ---- classes
    def add_class(self, name, color=None) -> int:
        for i, c in enumerate(self.classes):
            if c["name"] == name:
                return i
        color = color or PALETTE[len(self.classes) % len(PALETTE)]
        self.classes.append({"name": name, "color": color})
        self.dirty = True
        return len(self.classes) - 1

    def class_names(self):
        return [c["name"] for c in self.classes]

    def remove_class(self, idx):
        self.classes.pop(idx)
        for im in self.images:
            if im.tag is not None:
                im.tag = None if im.tag == idx else (im.tag - 1 if im.tag > idx else im.tag)
            im.anns = [a for a in im.anns if a.cls != idx]
            for a in im.anns:
                if a.cls > idx:
                    a.cls -= 1
        self.dirty = True

    # ---- images
    def prepare_images(self, paths, progress=None) -> list:
        """Записи для новых картинок (читает размеры файлов, проект не меняет - можно вызывать из потока).
        progress(i, n) -> False прерывает."""
        known = {os.path.normcase(i.path) for i in self.images}
        out = []
        paths = list(paths)
        for i, p in enumerate(paths):
            if progress is not None and i % 50 == 0 and progress(i, len(paths)) is False:
                break
            p = str(Path(p).resolve())
            k = os.path.normcase(p)
            if k in known or not imgio.is_image(p):
                continue
            try:
                w, h = imgio.size_of(p)
            except Exception:
                continue
            out.append(ImageEntry(p, w, h))
            known.add(k)
        return out

    def add_prepared(self, entries) -> int:
        if entries:
            self.images.extend(entries)
            self.images.sort(key=lambda i: i.path.lower())
            self.dirty = True
        return len(entries)

    def add_images(self, paths, progress=None) -> int:
        return self.add_prepared(self.prepare_images(paths, progress))

    # ---- persistence
    def write_to(self, target):
        """Write the project to `target` atomically without touching path / dirty (used for recovery copies)."""
        target = Path(target)
        base = target.parent
        data = {
            "version": 1,
            "name": self.name,
            "classes": self.classes,
            "presets": self.presets,
            "export": self.export,
            "keypoints": self.keypoints, "skeleton": self.skeleton, "flip_pairs": self.flip_pairs,
            "images": [{
                "path": imgio.rel_or_abs(Path(i.path), base).replace("\\", "/"), "w": i.w, "h": i.h,
                "reviewed": i.reviewed, "anns": [a.to_dict() for a in i.anns],
                **({"tag": i.tag} if i.tag is not None else {}),
            } for i in self.images],
        }
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, target)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def save(self, path=None):
        target = Path(path) if path else self.path
        assert target, "no path"
        if target.exists():                         # предыдущая версия сохраняется как резервная копия
            try:
                json.loads(textio.read_text(target))
                shutil.copy2(target, target.with_name(target.name + ".bak"))
            except (OSError, ValueError):               # файл не читается: хорошую .bak не затираем
                pass
        self.write_to(target)
        self.path = target          # only after a successful write
        self.dirty = False

    @staticmethod
    def load(path):
        path = Path(path)
        recovered = False
        try:
            data = json.loads(textio.read_text(path))
            if not isinstance(data, dict) or not isinstance(data.get("images", []), list):
                raise ValueError("не проект ML IDE")
        except (ValueError, UnicodeDecodeError):
            bak = path.with_name(path.name + ".bak")
            if not bak.exists():
                raise
            data = json.loads(textio.read_text(bak))      # файл повреждён - берём резервную копию
            recovered = True
        pr = Project(data.get("name", path.stem))
        pr.recovered_from_backup = recovered
        pr.path = path
        pr.classes = data.get("classes", [])
        pr.presets = data.get("presets", {})
        pr.export = data.get("export", {})
        pr.keypoints = data.get("keypoints", [])
        pr.skeleton = data.get("skeleton", [])
        pr.flip_pairs = data.get("flip_pairs", [])
        for d in data.get("images", []):
            p = Path(str(d["path"]).replace("\\", "/") if not Path(d["path"]).is_absolute() else d["path"])
            if not p.is_absolute():
                p = Path(os.path.normpath(os.path.join(path.parent.resolve(), p)))
            pr.images.append(ImageEntry(str(p), d["w"], d["h"],
                                        [Annotation.from_dict(a) for a in d.get("anns", [])],
                                        d.get("reviewed", False), d.get("tag")))
        pr.dirty = False
        return pr
