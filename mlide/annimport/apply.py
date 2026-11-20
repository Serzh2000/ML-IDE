"""Запись разобранной разметки в проект: поиск картинок, перевод координат, сопоставление классов."""
from __future__ import annotations

import math
import os
from pathlib import Path

from .. import imgio
from ..model import Annotation, ImageEntry, Project
from .resolve import ImageResolver
from .types import ApplyReport, Options, Parsed


def _clamp(pts, w, h):
    return [[min(max(float(x), 0.0), w), min(max(float(y), 0.0), h)] for x, y in pts]


def _finite(pts) -> bool:
    return all(math.isfinite(float(v)) for p in pts for v in p)


def _same(a, kind, cls, pts, tol: float = 1.0) -> bool:
    """Такая же разметка уже есть (тот же класс, вид и координаты с точностью до пикселя): повторный импорт не удваивает объекты."""
    return a.cls == cls and a.kind == kind and len(a.pts) == len(pts) and all(
        abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol for p, q in zip(a.pts, pts))


def name_of(c, parsed: Parsed) -> str:
    """Класс из источника -> имя: число расшифровывается через class_names, иначе берётся как есть."""
    names = parsed.class_names or []
    if isinstance(c, bool):
        return str(c)
    if isinstance(c, (int, float)) or (isinstance(c, str) and c.strip().lstrip("-").isdigit() and names):
        i = int(c)
        return str(names[i]) if 0 <= i < len(names) else f"class_{i}"
    return str(c)


def class_keys(parsed: Parsed) -> list[str]:
    """Имена классов, которые встречаются в разметке (для таблицы сопоставления классов)."""
    seen: dict[str, None] = {}
    for r in parsed.records:
        for o in r.objs:
            if o.cls is not None:
                seen.setdefault(name_of(o.cls, parsed), None)
        if r.tag is not None:
            seen.setdefault(name_of(r.tag, parsed), None)
    for n in parsed.class_names or []:
        seen.setdefault(str(n), None)
    return list(seen)


def apply_parsed(project: Project, parsed: Parsed, resolver: ImageResolver, opt: Options, progress=None) -> ApplyReport:
    rep = ApplyReport(warnings=list(parsed.notes))
    class_ids: dict[str, int] = {}

    def cls_idx(name: str):
        target = opt.class_map.get(name, name)
        if target is None:                       # класс отключён пользователем
            return None
        if target not in class_ids:
            before = len(project.classes)
            class_ids[target] = project.add_class(str(target))
            if len(project.classes) > before:
                rep.classes_added.append(str(target))
        return class_ids[target]

    if parsed.keypoints and not project.keypoints:
        project.keypoints = list(parsed.keypoints)
        project.skeleton = [list(p) for p in (parsed.skeleton or [])]
    n = len(parsed.records)
    handled: set[str] = set()
    index = {os.path.normcase(e.path): e for e in project.images}
    added_new = False
    for i, rec in enumerate(parsed.records):
        if progress and progress(i, n, rec.ref) is False:
            break
        path, st = resolver.find(rec.ref, rec.extra_refs)
        if path is None:
            rep.unmatched.append(rec.ref)
            continue
        if st == "ambiguous":
            rep.ambiguous.append(rec.ref)
        full = str(Path(path).resolve())                        # как хранит проект: полный путь (короткие имена 8.3 раскрываются)
        key = os.path.normcase(full)
        existing = index.get(key)
        if existing is None:
            entry = None
            if opt.add_missing_images and imgio.is_image(full):
                try:
                    w0, h0 = imgio.size_of(full)
                    entry = ImageEntry(full, w0, h0)
                except Exception:  # noqa
                    entry = None
            if entry is None:
                rep.unmatched.append(rec.ref)
                continue
            project.images.append(entry)                        # сортировка один раз в конце
            index[key] = entry
            added_new = True
            rep.images_added += 1
            is_new = True
        else:
            entry, is_new = existing, False
            if key not in handled:                   # повторная запись той же картинки из источника всегда дописывается
                if entry.anns and opt.existing == "skip":
                    rep.images_skipped += 1
                    continue
                if entry.anns and opt.existing == "replace":
                    entry.anns = []
        handled.add(key)
        w, h = entry.w, entry.h
        sx = (w / rec.width) if (rec.width and abs(rec.width - w) > 1) else 1.0     # набор уменьшали после разметки
        sy = (h / rec.height) if (rec.height and abs(rec.height - h) > 1) else 1.0
        added = 0
        for o in rec.objs:
            if o.cls is None:
                continue
            ci = cls_idx(name_of(o.cls, parsed))
            if ci is None:
                continue
            pts = [[(x * w if o.norm else x * sx), (y * h if o.norm else y * sy)] for x, y in o.pts]
            if not _finite(pts):                                  # NaN/inf из кривого файла: объект пропускаем
                continue
            if o.kind in ("box", "pose"):
                if len(pts) < 2:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                pts = _clamp([[min(xs), min(ys)], [max(xs), max(ys)]], w, h)
                if pts[1][0] - pts[0][0] < 1 or pts[1][1] - pts[0][1] < 1:
                    continue
            else:
                if len(pts) < 3:
                    continue
                pts = _clamp(pts, w, h)
            kps = None
            if o.kind == "pose" and o.kps:
                kps = [[(k[0] * w if o.norm else k[0] * sx), (k[1] * h if o.norm else k[1] * sy), int(k[2]) if len(k) > 2 else 2]
                       for k in o.kps]
            kind = o.kind if o.kind in ("box", "poly", "obb", "pose") else "poly"
            if not is_new and entry.anns and any(_same(a, kind, ci, pts) for a in entry.anns):
                continue
            entry.anns.append(Annotation(ci, kind, pts, kps=kps, conf=o.conf))
            added += 1
        rep.objects_added += added
        if opt.import_tags and rec.tag is not None:
            ti = cls_idx(name_of(rec.tag, parsed))
            if ti is not None:
                entry.tag = ti
                rep.tags_set += 1
        if not is_new and (added or rec.tag is not None):
            rep.images_updated += 1
    if added_new:
        project.images.sort(key=lambda i: i.path.lower())
    if rep.images_added or rep.images_updated or rep.classes_added:
        project.dirty = True
    return rep
