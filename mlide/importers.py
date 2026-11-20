"""Dataset import: COCO instances JSON and Pascal VOC XML (no Qt)."""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from . import imgio, textio
from .model import Annotation, ImageEntry, Project


@dataclass
class ImportReport:
    images_added: int = 0
    annotations_added: int = 0
    skipped_existing: int = 0
    missing_files: int = 0
    classes_added: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        s = (f"Добавлено изображений: {self.images_added}, объектов: {self.annotations_added}")
        if self.classes_added:
            s += f", новых классов: {len(self.classes_added)}"
        if self.skipped_existing:
            s += f". Уже в проекте (пропущено): {self.skipped_existing}"
        if self.missing_files:
            s += f". Файлы не найдены: {self.missing_files}"
        if self.warnings:
            s += f". Предупреждений: {len(self.warnings)}"
        return s + "."


def _key(p) -> str:
    return os.path.normcase(str(Path(p).resolve()))


def _build_lookup(roots) -> dict:
    """basename (lowercase) -> first full path, over all given roots recursively."""
    look = {}
    for r in roots:
        r = Path(r)
        if not r.is_dir():
            continue
        for dp, _dn, fns in os.walk(r):
            for fn in fns:
                if imgio.is_image(fn):
                    look.setdefault(fn.lower(), os.path.join(dp, fn))
    return look


def _add_class(project: Project, name: str, report: ImportReport) -> int:
    before = len(project.classes)
    idx = project.add_class(name)
    if len(project.classes) > before:
        report.classes_added.append(name)
    return idx


def _find_entry(project: Project, path) -> ImageEntry | None:
    k = _key(path)
    for e in project.images:
        if os.path.normcase(e.path) == k:
            return e
    return None


def _clamp_pts(pts, w, h):
    return [[min(max(float(x), 0.0), w), min(max(float(y), 0.0), h)] for x, y in pts]


def _poly_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def _register_image(project: Project, path, report: ImportReport):
    """Returns (entry, is_new) or (None, False) if unreadable."""
    existing = _find_entry(project, path)
    if existing is not None:
        return existing, False
    if not project.add_images([path]):
        return None, False
    return _find_entry(project, path), True


# ----------------------------------------------------------------------------- COCO
def import_coco(project: Project, json_path, images_dir=None, progress=None) -> ImportReport:
    rep = ImportReport()
    jp = Path(json_path)
    data = json.loads(textio.read_text(jp))
    base = jp.parent
    search_roots = []
    if images_dir:
        search_roots.append(Path(images_dir))
    search_roots += [base / "images", base]
    lookup = None

    cats = {}
    for c in data.get("categories", []):
        cats[c["id"]] = _add_class(project, str(c["name"]), rep)

    anns_by_img: dict = {}
    for a in data.get("annotations", []):
        anns_by_img.setdefault(a.get("image_id"), []).append(a)

    images = data.get("images", [])
    n = len(images)
    for i, im in enumerate(images):
        if progress:
            if progress(i, n, str(im.get("file_name", ""))) is False:
                break
        fname = str(im.get("file_name", ""))
        found = None
        rel = Path(fname)
        for r in search_roots:
            for cand in ((r / rel), (r / rel.name)):
                if cand.is_file():
                    found = cand
                    break
            if found:
                break
        if found is None:
            if lookup is None:
                lookup = _build_lookup(search_roots)
            f2 = lookup.get(rel.name.lower())
            found = Path(f2) if f2 else None
        if found is None:
            rep.missing_files += 1
            continue
        entry, is_new = _register_image(project, found, rep)
        if entry is None:
            rep.warnings.append(f"Не удалось прочитать: {fname}")
            continue
        if not is_new:
            rep.skipped_existing += 1
            continue
        rep.images_added += 1
        w, h = entry.w, entry.h
        for a in anns_by_img.get(im.get("id"), []):
            if a.get("iscrowd"):
                continue
            if a.get("category_id") not in cats:
                rep.warnings.append(f"Неизвестная категория {a.get('category_id')}")
                continue
            cls = cats[a["category_id"]]
            seg = a.get("segmentation")
            polys = []
            if isinstance(seg, list):
                for s in seg:
                    if isinstance(s, list) and len(s) >= 6:
                        polys.append([[s[j], s[j + 1]] for j in range(0, len(s) - 1, 2)])
            if polys:
                best = max(polys, key=_poly_area)
                entry.anns.append(Annotation(cls, "poly", _clamp_pts(best, w, h)))
                rep.annotations_added += 1
                continue
            if isinstance(seg, dict):       # RLE
                rep.warnings.append("RLE-маска пропущена")
                continue
            bb = a.get("bbox")
            if bb and len(bb) == 4:
                x, y, bw, bh = [float(v) for v in bb]
                pts = _clamp_pts([[x, y], [x + bw, y + bh]], w, h)
                if pts[1][0] - pts[0][0] < 1e-6 or pts[1][1] - pts[0][1] < 1e-6:
                    continue
                entry.anns.append(Annotation(cls, "box", pts))
                rep.annotations_added += 1
    if rep.images_added or rep.classes_added:
        project.dirty = True
    return rep


# ----------------------------------------------------------------------------- VOC
def import_voc(project: Project, root_dir, progress=None) -> ImportReport:
    rep = ImportReport()
    root = Path(root_dir)
    ann_dir = root / "Annotations"
    xmls = sorted(ann_dir.glob("*.xml")) if ann_dir.is_dir() else sorted(root.glob("*.xml"))
    if not xmls:
        xmls = sorted(root.rglob("*.xml"))
    roots = [root]
    lookup = None
    n = len(xmls)
    for i, xp in enumerate(xmls):
        if progress:
            if progress(i, n, xp.name) is False:
                break
        try:
            tree = ET.parse(str(xp))
        except (ET.ParseError, OSError):
            rep.warnings.append(f"Не удалось разобрать XML: {xp.name}")
            continue
        r = tree.getroot()
        if r.tag != "annotation":
            continue
        fname = (r.findtext("filename") or "").strip()
        cands = []
        pth = (r.findtext("path") or "").strip()
        if fname:
            cands += [root / "JPEGImages" / fname, xp.parent / fname, root / fname]
        if pth:
            cands.append(Path(pth))
            cands.append(root / "JPEGImages" / Path(pth.replace("\\", "/")).name)
        if not fname:
            for ext in imgio.IMAGE_EXTS:
                cands += [xp.parent / (xp.stem + ext), root / "JPEGImages" / (xp.stem + ext)]
        found = next((c for c in cands if c.is_file()), None)
        if found is None:
            if lookup is None:
                lookup = _build_lookup(roots)
            nm = (fname or (xp.stem + ".jpg"))
            f2 = lookup.get(Path(nm.replace("\\", "/")).name.lower())
            if f2 is None:
                for ext in imgio.IMAGE_EXTS:
                    f2 = lookup.get((xp.stem + ext).lower())
                    if f2:
                        break
            found = Path(f2) if f2 else None
        if found is None:
            rep.missing_files += 1
            continue
        entry, is_new = _register_image(project, found, rep)
        if entry is None:
            rep.warnings.append(f"Не удалось прочитать: {found.name}")
            continue
        if not is_new:
            rep.skipped_existing += 1
            continue
        rep.images_added += 1
        for o in r.findall("object"):
            name = (o.findtext("name") or "").strip()
            bb = o.find("bndbox")
            if not name or bb is None:
                continue
            try:
                xmin, ymin, xmax, ymax = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError):
                rep.warnings.append(f"Некорректный bndbox в {xp.name}")
                continue
            cls = _add_class(project, name, rep)
            pts = _clamp_pts([[xmin - 1, ymin - 1], [xmax, ymax]], entry.w, entry.h)
            if pts[1][0] - pts[0][0] < 1e-6 or pts[1][1] - pts[0][1] < 1e-6:
                continue
            entry.anns.append(Annotation(cls, "box", pts))
            rep.annotations_added += 1
    if rep.images_added or rep.classes_added:
        project.dirty = True
    return rep


# ----------------------------------------------------------------------------- классификация: папки = классы
def import_class_folders(project: Project, root_dir, progress=None) -> ImportReport:
    """Изображения лежат в папках, имя папки - класс (train/cat/1.jpg, val/dog/2.jpg или cat/1.jpg).
    Каждому изображению присваивается класс кадра (tag). Папки train/val/test в названия классов не попадают."""
    from . import imgio
    rep = ImportReport()
    root = Path(root_dir)
    split_names = {"train", "val", "valid", "validation", "test", "training"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and imgio.is_image(p))
    classes_found = sorted({p.parent.name for p in files if p.parent.name.lower() not in split_names and p.parent != root},
                           key=lambda s: (len(s), s) if s.isdigit() else (0, s.lower()))
    if not classes_found:
        rep.warnings.append("Не найдено подпапок с изображениями: ожидается структура класс/изображение.")
        return rep
    for i, p in enumerate(files):
        if progress and progress(i, len(files), p.name) is False:
            break
        name = p.parent.name
        if name.lower() in split_names or p.parent == root:
            continue
        idx = _add_class(project, name, rep)
        e, is_new = _register_image(project, p, rep)
        if e is None:
            rep.missing_files += 1
            continue
        if is_new:
            rep.images_added += 1
        elif e.tag is not None:
            rep.skipped_existing += 1
            continue
        e.tag = idx
        rep.annotations_added += 1
    project.dirty = True
    return rep
