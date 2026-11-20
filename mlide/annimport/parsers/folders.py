"""Классификация: имя папки с картинками = класс кадра (cat/1.jpg, dog/2.jpg; допустимы train/val/test сверху)."""
from __future__ import annotations

import os

from ... import imgio
from ..types import Detection, Parsed, Record

NAME = "folders"
TITLE = {"ru": "Папки = классы (классификация)", "en": "Folders = classes (classification)"}
STRUCT = {"train", "val", "valid", "validation", "test", "testing", "training", "images", "image", "imgs", "img", "labels", "data",
          "dataset", "datasets", "raw", "all", "jpegimages", "annotations", "photos", "pictures", "pics", "output", "input"}


def _scan(ctx, limit=400_000):
    """-> {имя_папки_класса: [полные пути картинок]}; папки train/val/images и подобные не считаются классами."""
    by: dict[str, list[str]] = {}
    n = 0
    for root in ctx.image_roots():
        for dp, dn, fns in os.walk(root):
            dn[:] = [d for d in dn if not d.startswith(".")]
            cls = os.path.basename(dp)
            for fn in fns:
                if not imgio.is_image(fn):
                    continue
                n += 1
                if cls.lower() not in STRUCT and os.path.normcase(dp) != os.path.normcase(str(root)):
                    by.setdefault(cls, []).append(os.path.join(dp, fn))
            if n > limit:
                break
    return by


def sniff(ctx):
    by = _scan(ctx)
    classes = [c for c, v in by.items() if v]
    total = sum(len(v) for v in by.values())
    if len(classes) < 2 or total < 4:
        return None
    return Detection(NAME, 0.45, [], f"{len(classes)} папок-классов, {total} картинок", {"classes": len(classes)})


def parse(ctx, det, opt) -> Parsed:
    by = _scan(ctx)
    pd = Parsed("folders")
    for cls in sorted(by, key=lambda s: (len(s), s) if s.isdigit() else (0, s.lower())):
        for p in by[cls]:
            pd.records.append(Record(ref=p, tag=cls))
    return pd
