"""Поимённые .txt с разметкой НЕ в формате YOLO: DOTA (углы), KITTI, пиксельные «класс x1 y1 x2 y2» / «x1 y1 x2 y2 класс»."""
from __future__ import annotations

import math
import re
from pathlib import Path

from ..types import Detection, Obj, Parsed, Record
from .voc import Notes

NAME = "txtper"
TITLE = {"ru": "Текстовые файлы разметки (DOTA / KITTI / x1 y1 x2 y2)", "en": "Per-image text labels (DOTA / KITTI / x1 y1 x2 y2)"}

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_SKIP_STEMS = {"classes", "obj", "names", "labels", "train", "val", "valid", "test", "trainval", "label_map", "notes"}
_CLASS_FILES = ("classes.txt", "obj.names", "names.txt", "labels.txt", "classes.names", "class.txt", "classes.names.txt")
_HEADERS = ("imagesource:", "gsd:")
_SAMPLE = 30
_PEEK = 20000


def isnum(t: str) -> bool:
    return bool(_NUM.match(t))


def _isint(t: str) -> bool:
    return isnum(t) and float(t) == int(float(t))


def _is_label_candidate(p: Path) -> bool:
    s = p.stem.lower()
    return not (s in _SKIP_STEMS or s.startswith("readme") or s.startswith("license") or s.startswith("requirements"))


def _tokens(text: str):
    out = []
    for ln in text.splitlines():
        ln = ln.strip().lstrip("﻿")
        if not ln or ln.startswith("#") or ln.lower().startswith(_HEADERS):
            continue
        out.append(ln.split())
    return out


def _group(t):
    """Какой раскладке может принадлежать строка: dota | kitti | xy | None."""
    n = len(t)
    if n in (9, 10) and all(isnum(x) for x in t[:8]) and not isnum(t[8]) and (n == 9 or isnum(t[9])):
        return "dota"
    if n in (15, 16) and not isnum(t[0]) and all(isnum(x) for x in t[1:15]):
        return "kitti"
    if n == 5 and (all(isnum(x) for x in t[1:]) or all(isnum(x) for x in t[:4])):
        return "xy"
    return None


def _decide_xy(lines):
    """Строки по 5 токенов -> параметры или None (в том числе для YOLO)."""
    first_ok = all(all(isnum(x) for x in t[1:]) for t in lines)
    last_ok = all(all(isnum(x) for x in t[:4]) for t in lines)
    cands = {}
    for pos, ok in (("first", first_ok), ("last", last_ok)):
        if not ok:
            continue
        co = [[float(x) for x in (t[1:] if pos == "first" else t[:4])] for t in lines]
        if all(c[2] > c[0] and c[3] > c[1] for c in co):
            mode = "xyxy"
        elif all(c[2] > 0 and c[3] > 0 for c in co):
            mode = "xywh"
        else:
            continue
        ctoks = [t[0] if pos == "first" else t[4] for t in lines]
        cands[pos] = (mode, co, ctoks)
    if not cands:
        return None
    if len(cands) == 2:
        c1 = all(_isint(t) for t in cands["first"][2])
        c2 = all(_isint(t) for t in cands["last"][2])
        if c1 != c2:
            pos = "first" if c1 else "last"
        elif not c1:
            return None
        else:
            m1 = max(float(t) for t in cands["first"][2])
            m2 = max(float(t) for t in cands["last"][2])
            pos = "first" if m1 <= m2 else "last"
    else:
        pos = next(iter(cands))
    mode, co, ctoks = cands[pos]
    flat = [v for c in co for v in c]
    pixel = max(flat) > 1.5 or all(v == int(v) for v in flat)
    cls_numeric_int = all(_isint(t) for t in ctoks)
    if not pixel and pos == "first" and cls_numeric_int:
        return None                                   # нормированный «0 0.5 0.5 0.2 0.3» - это YOLO, его читает другой разборщик
    if not pixel and all(isnum(t) for t in ctoks) and not cls_numeric_int:
        pass
    return {"layout": "xy", "cls_pos": pos, "mode": mode, "norm": not pixel}


def _sample(ctx, cands):
    step = max(1, len(cands) // (_SAMPLE * 2))
    out = []
    for p in cands[::step][: _SAMPLE * 2]:
        try:
            text = ctx.read_text(p, _PEEK)
        except OSError:
            continue
        lines = _tokens(text)
        if len(text) >= _PEEK // 2 and lines:
            lines = lines[:-1]                        # последняя строка могла быть обрезана
        if lines:
            out.append(lines[:200])
        if len(out) >= _SAMPLE:
            break
    return out


def sniff(ctx):
    cands = [p for p in ctx.files(".txt") if _is_label_candidate(p)]
    if not cands:
        return None
    sample = _sample(ctx, cands)
    if not sample:
        return None
    groups = {}
    for lines in sample:
        gs = {_group(t) for t in lines}
        g = next(iter(gs)) if len(gs) == 1 else None
        groups.setdefault(g, []).append(lines)
    best = max((g for g in groups if g), key=lambda g: len(groups[g]), default=None)
    if best is None or len(groups[best]) < math.ceil(0.7 * len(sample)):
        return None
    lines = [t for fl in groups[best] for t in fl]
    if best == "xy":
        params = _decide_xy(lines)
        if not params:
            return None
        word = any(not isnum(t[0 if params["cls_pos"] == "first" else 4]) for t in lines)
        score = 0.65 if word else 0.4
        desc = "x1 y1 x2 y2" if params["mode"] == "xyxy" else "x y w h"
        desc = ("класс " + desc) if params["cls_pos"] == "first" else (desc + " класс")
        desc += ", доли кадра" if params["norm"] else ", пиксели"
    elif best == "dota":
        params, score, desc = {"layout": "dota"}, 0.88, "DOTA: 4 угла и класс"
    else:
        params, score, desc = {"layout": "kitti"}, 0.92, "KITTI"
    n = len(cands)
    return Detection(NAME, score, list(cands), f"{n} файлов .txt ({desc})", params)


def _classes_from(folder: Path, ctx):
    for d in (folder, folder.parent):
        for fn in _CLASS_FILES:
            f = d / fn
            if f.is_file():
                try:
                    names = [ln.strip() for ln in ctx.read_text(f).splitlines() if ln.strip()]
                except OSError:
                    continue
                if names:
                    return names
        for fn in ("data.yaml", "dataset.yaml", "data.yml"):
            f = d / fn
            if f.is_file():
                try:
                    import yaml
                    nm = (yaml.safe_load(ctx.read_text(f)) or {}).get("names")
                except Exception:  # noqa
                    continue
                if isinstance(nm, dict):
                    try:
                        return [str(nm[k]) for k in sorted(nm, key=int)]
                    except (ValueError, TypeError):
                        continue
                if isinstance(nm, list) and nm:
                    return [str(x) for x in nm]
    return None


def _cls(tok: str):
    return int(float(tok)) if _isint(tok) else tok


def _line_obj(t, prm):
    lay = prm["layout"]
    if lay == "dota":
        if _group(t) != "dota":
            raise ValueError
        v = [float(x) for x in t[:8]]
        return Obj("obb", t[8], [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]], [v[6], v[7]]])
    if lay == "kitti":
        if _group(t) != "kitti":
            raise ValueError
        if t[0].lower() == "dontcare":
            return None
        v = [float(x) for x in t[4:8]]
        conf = float(t[15]) if len(t) == 16 else None
        return Obj("box", t[0], [[v[0], v[1]], [v[2], v[3]]], conf=conf)
    if len(t) != 5:
        raise ValueError
    first = prm["cls_pos"] == "first"
    ctok = t[0] if first else t[4]
    v = [float(x) for x in (t[1:] if first else t[:4])]
    if prm["mode"] == "xywh":
        v = [v[0], v[1], v[0] + v[2], v[1] + v[3]]
    return Obj("box", _cls(ctok), [[v[0], v[1]], [v[2], v[3]]], norm=prm["norm"])


def parse(ctx, det, opt):
    out = Parsed("txtper")
    notes = Notes(out.notes)
    prm = det.params or {"layout": "kitti"}
    numeric = False
    for p in det.files:
        try:
            text = ctx.read_text(p)
        except OSError:
            notes.add(f"Не удалось прочитать: {Path(p).name}")
            continue
        rec = Record(Path(p).stem)
        bad = good = 0
        for t in _tokens(text):
            try:
                o = _line_obj(t, prm)
            except (ValueError, IndexError):
                bad += 1
                continue
            good += 1
            if o is not None:
                rec.objs.append(o)
                numeric = numeric or isinstance(o.cls, int)
        if bad and not good:
            continue                                   # не наш файл
        if bad:
            notes.add(f"{Path(p).name}: пропущено некорректных строк - {bad}")
        out.records.append(rec)
    if numeric and det.files:
        out.class_names = _classes_from(Path(det.files[0]).parent, ctx)
    notes.close()
    return out
