"""Pascal VOC / LabelImg: по одному .xml на картинку (корень <annotation>)."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..types import Detection, Obj, Parsed, Record

NAME = "voc"
TITLE = {"ru": "Pascal VOC / LabelImg (XML)", "en": "Pascal VOC / LabelImg (XML)"}

_ROOT_RE = re.compile(r"<([A-Za-z_][\w.\-]*)")
_CLEAN_RE = re.compile(r"<\?.*?\?>|<!--.*?-->|<!DOCTYPE[^>]*>", re.S)
_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)


def root_tag(text: str):
    m = _ROOT_RE.search(_CLEAN_RE.sub("", text))
    return m.group(1) if m else None


def load_xml(text: str):
    text = text.lstrip("﻿")
    text = _DECL_RE.sub("", text, count=1)
    return ET.fromstring(text.strip())


class Notes:
    def __init__(self, lst, cap=20):
        self.lst, self.cap, self.extra = lst, cap, 0

    def add(self, msg):
        if len(self.lst) < self.cap:
            self.lst.append(msg)
        else:
            self.extra += 1

    def close(self):
        if self.extra:
            self.lst.append(f"... и ещё предупреждений: {self.extra}")


def sniff(ctx):
    xmls = ctx.files(".xml")
    if not xmls:
        return None
    step = max(1, len(xmls) // 12)
    sample = xmls[::step][:12]
    hits = objs = 0
    for p in sample:
        try:
            head = ctx.read_text(p, 6000)
        except OSError:
            continue
        if root_tag(head) == "annotation":
            hits += 1
            if "<object" in head or "<size" in head or "<filename" in head:
                objs += 1
    if not hits or hits < 0.5 * len(sample) or not objs:
        return None
    score = 0.75
    if any(p.parent.name.lower() == "annotations" for p in sample):
        score += 0.1
    if hits == len(sample):
        score += 0.05
    n = len(xmls) if hits == len(sample) else int(len(xmls) * hits / len(sample))
    return Detection(NAME, min(score, 0.95), list(xmls), f"{n} файлов Pascal VOC (XML)")


def _f(el, tag, default=None):
    if el is None:
        return default
    t = el.findtext(tag)
    if t is None:
        return default
    try:
        return float(t.strip())
    except ValueError:
        return default


def _obb(cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [[cx + x * c - y * s, cy + x * s + y * c] for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def _poly(poly):
    pts = []
    for pt in poly.findall("pt"):
        x, y = _f(pt, "x"), _f(pt, "y")
        if x is not None and y is not None:
            pts.append([x, y])
    if pts:
        return pts
    i = 1
    while True:
        x, y = _f(poly, f"x{i}"), _f(poly, f"y{i}")
        if x is None or y is None:
            break
        pts.append([x, y])
        i += 1
    return pts


def _record(root, xp: Path):
    fname = (root.findtext("filename") or "").strip()
    pth = (root.findtext("path") or "").strip()
    base = Path(pth.replace("\\", "/")).name if pth else ""
    ref = fname or pth or xp.stem
    extra = [x for x in (base, pth if fname else "", xp.stem) if x and x != ref]
    sz = root.find("size")
    w, h = _f(sz, "width"), _f(sz, "height")
    rec = Record(ref, width=int(w) if w else None, height=int(h) if h else None, extra_refs=extra)
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        if not name:
            continue
        bb = o.find("bndbox")
        rb = o.find("robndbox")
        pg = o.find("polygon")
        if rb is not None:
            cx, cy, bw, bh, a = _f(rb, "cx"), _f(rb, "cy"), _f(rb, "w"), _f(rb, "h"), _f(rb, "angle", 0.0)
            if None not in (cx, cy, bw, bh):
                rec.objs.append(Obj("obb", name, _obb(cx, cy, bw, bh, a)))
        elif pg is not None:
            pts = _poly(pg)
            if len(pts) >= 3:
                rec.objs.append(Obj("poly", name, pts))
        elif bb is not None:
            x1, y1, x2, y2 = (_f(bb, k) for k in ("xmin", "ymin", "xmax", "ymax"))
            if None in (x1, y1, x2, y2):
                continue
            if x1 >= 1:
                x1 -= 1
            if y1 >= 1:
                y1 -= 1
            rec.objs.append(Obj("box", name, [[x1, y1], [x2, y2]]))
    return rec


def parse(ctx, det, opt):
    out = Parsed("voc")
    notes = Notes(out.notes)
    for xp in det.files:
        try:
            root = load_xml(ctx.read_text(xp))
        except (ET.ParseError, OSError, ValueError):
            notes.add(f"Не удалось разобрать XML: {Path(xp).name}")
            continue
        if root.tag != "annotation":
            continue
        try:
            out.records.append(_record(root, Path(xp)))
        except Exception:  # noqa
            notes.add(f"Ошибка в файле: {Path(xp).name}")
    notes.close()
    return out
