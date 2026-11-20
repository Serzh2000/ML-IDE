"""CVAT for images 1.1: один XML с <image name width height> и разметкой внутри."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..types import Detection, Obj, Parsed, Record
from .voc import Notes, load_xml, root_tag

NAME = "cvat"
TITLE = {"ru": "CVAT for images 1.1 (XML)", "en": "CVAT for images 1.1 (XML)"}


def sniff(ctx):
    cands = []
    for p in ctx.files(".xml"):
        try:
            head = ctx.read_text(p, 200_000)
        except OSError:
            continue
        if root_tag(head[:4000]) != "annotations":
            continue
        if re.search(r"<image\b", head):
            cands.append((p, 0.95 if "<meta" in head else 0.85))
        if len(cands) >= 200:
            break
    if not cands:
        return None
    return Detection(NAME, max(s for _, s in cands), [p for p, _ in cands], f"{len(cands)} файл(ов) CVAT for images (XML)")


def _rot_box(xtl, ytl, xbr, ybr, deg):
    cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[cx + x * c - y * s, cy + x * s + y * c] for x, y in
            ((xtl - cx, ytl - cy), (xbr - cx, ytl - cy), (xbr - cx, ybr - cy), (xtl - cx, ybr - cy))]


def _fl(el, k, d=0.0):
    try:
        return float(el.get(k))
    except (TypeError, ValueError):
        return d


def _pts(s):
    out = []
    for part in (s or "").replace(" ", ";").split(";"):
        if not part.strip():
            continue
        xy = part.split(",")
        out.append([float(xy[0]), float(xy[1])])
    return out


def parse(ctx, det, opt):
    out = Parsed("cvat")
    notes = Notes(out.notes)
    skipped = {}
    for xp in det.files:
        try:
            root = load_xml(ctx.read_text(xp))
        except (ET.ParseError, OSError, ValueError):
            notes.add(f"Не удалось разобрать XML: {Path(xp).name}")
            continue
        if root.tag != "annotations":
            continue
        for im in root.findall("image"):
            name = im.get("name")
            if not name:
                continue
            w, h = im.get("width"), im.get("height")
            try:
                rec = Record(name, width=int(float(w)) if w else None, height=int(float(h)) if h else None)
            except ValueError:
                rec = Record(name)
            for el in im:
                lab = el.get("label")
                try:
                    if el.tag == "box" and lab:
                        x1, y1, x2, y2 = (_fl(el, k) for k in ("xtl", "ytl", "xbr", "ybr"))
                        rot = _fl(el, "rotation")
                        if abs(rot % 360) > 1e-6 and abs(rot % 360 - 360) > 1e-6:
                            rec.objs.append(Obj("obb", lab, _rot_box(x1, y1, x2, y2, rot)))
                        else:
                            rec.objs.append(Obj("box", lab, [[x1, y1], [x2, y2]]))
                    elif el.tag == "polygon" and lab:
                        pts = _pts(el.get("points"))
                        if len(pts) >= 3:
                            rec.objs.append(Obj("poly", lab, pts))
                    elif el.tag == "tag" and lab:
                        if rec.tag is None:
                            rec.tag = lab
                    elif el.tag in ("points", "polyline", "mask", "ellipse", "cuboid", "skeleton"):
                        skipped[el.tag] = skipped.get(el.tag, 0) + 1
                except (ValueError, IndexError):
                    notes.add(f"Некорректный элемент <{el.tag}> в кадре {name}")
            out.records.append(rec)
    for tag, n in skipped.items():
        notes.add(f"Пропущено элементов <{tag}>: {n} (не поддерживаются)")
    notes.close()
    return out
