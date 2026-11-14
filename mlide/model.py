"""Data model: annotations, images, classes, project (JSON on disk)."""
from __future__ import annotations

import copy
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import imgio

PALETTE = [
    "#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93", "#ff924c",
    "#52b788", "#f15bb5", "#00bbf9", "#9b5de5", "#fee440", "#00f5d4",
]


@dataclass
class Annotation:
    cls: int
    kind: str                      # "box" -> pts=[[x1,y1],[x2,y2]];  "poly" -> pts=[[x,y],...]
    pts: list
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    suggested: bool = False        # produced by auto-annotation, awaiting review
    conf: float | None = None

    def bbox(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return min(xs), min(ys), max(xs), max(ys)

    def polygon(self):
        if self.kind == "box":
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
        return d

    @staticmethod
    def from_dict(d):
        return Annotation(cls=int(d["cls"]), kind=d["kind"], pts=[list(p) for p in d["pts"]],
                          suggested=bool(d.get("suggested", False)), conf=d.get("conf"))


@dataclass
class ImageEntry:
    path: str                      # absolute path in memory
    w: int
    h: int
    anns: list = field(default_factory=list)
    reviewed: bool = False

    @property
    def name(self):
        return Path(self.path).name

    def snapshot(self):
        return copy.deepcopy([a.to_dict() for a in self.anns])

    def restore(self, snap):
        self.anns = [Annotation.from_dict(d) for d in snap]


class Project:
    def __init__(self, name="Без названия"):
        self.name = name
        self.path: Path | None = None
        self.classes: list[dict] = []
        self.images: list[ImageEntry] = []
        self.presets: dict[str, dict] = {}      # user augmentation presets
        self.export: dict = {}                  # last export settings
        self.dirty = False

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
            im.anns = [a for a in im.anns if a.cls != idx]
            for a in im.anns:
                if a.cls > idx:
                    a.cls -= 1
        self.dirty = True

    # ---- images
    def add_images(self, paths) -> int:
        known = {os.path.normcase(i.path) for i in self.images}
        added = 0
        for p in paths:
            p = str(Path(p).resolve())
            if os.path.normcase(p) in known or not imgio.is_image(p):
                continue
            try:
                w, h = imgio.size_of(p)
            except Exception:
                continue
            self.images.append(ImageEntry(p, w, h))
            known.add(os.path.normcase(p))
            added += 1
        if added:
            self.images.sort(key=lambda i: i.path.lower())
            self.dirty = True
        return added

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
            "images": [{
                "path": imgio.rel_or_abs(Path(i.path), base), "w": i.w, "h": i.h,
                "reviewed": i.reviewed, "anns": [a.to_dict() for a in i.anns],
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
        self.write_to(target)
        self.path = target          # only after a successful write
        self.dirty = False

    @staticmethod
    def load(path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        pr = Project(data.get("name", path.stem))
        pr.path = path
        pr.classes = data.get("classes", [])
        pr.presets = data.get("presets", {})
        pr.export = data.get("export", {})
        for d in data.get("images", []):
            p = Path(d["path"])
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            pr.images.append(ImageEntry(str(p), d["w"], d["h"],
                                        [Annotation.from_dict(a) for a in d.get("anns", [])],
                                        d.get("reviewed", False)))
        pr.dirty = False
        return pr
