"""VGG Image Annotator (VIA): JSON-проект/экспорт и CSV-экспорт."""
from __future__ import annotations

import csv
import io
import json

from ..types import Detection, Obj, Parsed, Record

NAME = "via"
TITLE = {"ru": "VGG Image Annotator (VIA)", "en": "VGG Image Annotator (VIA)"}

HEAD = 2_000_000
MAX_SNIFF = 30
MAX_NOTES = 20
CSV_COLS = {"filename", "region_shape_attributes", "region_attributes"}
PREFER = ("label", "class", "name", "type", "category", "object")


def _entries(d):
    """JSON VIA -> список записей-картинок (dict с filename/regions) или None."""
    if not isinstance(d, dict):
        return None
    if isinstance(d.get("_via_img_metadata"), dict):
        src = d["_via_img_metadata"]
    else:
        src = d
    out = [v for v in src.values() if isinstance(v, dict) and "filename" in v and "regions" in v]
    if not out:
        return None
    if src is d and len(out) < 0.5 * len(d):
        return None
    return out


def _csv_head_ok(txt: str) -> bool:
    first = txt.lstrip("﻿").split("\n", 1)[0]
    cols = {c.strip().strip('"').lower() for c in first.replace(";", ",").split(",")}
    return CSV_COLS <= cols


def sniff(ctx):
    files, best = [], 0.0
    for i, p in enumerate(ctx.files(".json", ".csv")):
        if i >= MAX_SNIFF:
            break
        try:
            txt = ctx.read_text(p, limit=HEAD)
        except OSError:
            continue
        if p.suffix.lower() == ".csv":
            if _csv_head_ok(txt[:4000]):
                files.append(p)
                best = max(best, 0.95)
        elif '"_via_img_metadata"' in txt or ('"regions"' in txt and '"filename"' in txt):
            d = ctx.read_json(p)
            if isinstance(d, dict) and isinstance(d.get("_via_img_metadata"), dict):
                files.append(p)
                best = max(best, 0.97)
            elif _entries(d):
                files.append(p)
                best = max(best, 0.9)
    if not files:
        return None
    return Detection(NAME, best, files, f"{len(files)} файл(ов) VIA")


def _val(v):
    """Значение атрибута -> строка класса (checkbox-словарь -> первый отмеченный)."""
    if isinstance(v, dict):
        on = [k for k, x in v.items() if x]
        return str(on[0]) if on else ""
    if v is None:
        return ""
    if isinstance(v, bool):
        return ""
    return str(v).strip()


def _pick_key(attr_dicts, via_attrs):
    keys: dict = {}
    for a in attr_dicts:
        for k, v in a.items():
            keys.setdefault(k, []).append(_val(v))
    keys = {k: v for k, v in keys.items() if any(v)}
    if not keys:
        return None
    if isinstance(via_attrs, dict):
        reg = via_attrs.get("region")
        if isinstance(reg, dict):
            for k, spec in reg.items():
                if k in keys and isinstance(spec, dict) and spec.get("type") in ("dropdown", "radio"):
                    return k
    low = {k.lower(): k for k in keys}
    for n in PREFER:
        if n in low:
            return low[n]
    if len(keys) == 1:
        return next(iter(keys))
    for k, vals in keys.items():
        vv = [x for x in vals if x]
        if len(set(vv)) < len(vv):
            return k
    return next(iter(keys))


def _shape(sa):
    """shape_attributes -> (kind, pts) или (None, причина)."""
    if not isinstance(sa, dict):
        return None, None
    name = str(sa.get("name") or "").lower()
    try:
        if name == "rect":
            x, y, w, h = (float(sa[k]) for k in ("x", "y", "width", "height"))
            return "box", [[x, y], [x + w, y + h]]
        if name == "polygon":
            xs, ys = sa["all_points_x"], sa["all_points_y"]
            pts = [[float(a), float(b)] for a, b in zip(xs, ys)]
            return ("poly", pts) if len(pts) >= 3 else (None, None)
        if name == "circle":
            cx, cy, r = float(sa["cx"]), float(sa["cy"]), float(sa["r"])
            return "box", [[cx - r, cy - r], [cx + r, cy + r]]
        if name == "ellipse":
            cx, cy, rx, ry = (float(sa[k]) for k in ("cx", "cy", "rx", "ry"))
            return "box", [[cx - rx, cy - ry], [cx + rx, cy + ry]]
    except (KeyError, TypeError, ValueError):
        return None, "bad"
    return None, name or None


class _Notes:
    def __init__(self, parsed):
        self.parsed, self.dropped = parsed, 0

    def add(self, msg):
        if len(self.parsed.notes) < MAX_NOTES:
            self.parsed.notes.append(msg)
        else:
            self.dropped += 1


def _build(items, via_attrs, fname, parsed, notes):
    """items: [(filename, regions[(shape_attrs, region_attrs)], file_attrs)]"""
    attr_dicts = [ra for _f, regs, _fa in items for _sa, ra in regs if isinstance(ra, dict)]
    key = _pick_key(attr_dicts, via_attrs)
    skipped: dict = {}
    for filename, regs, fattrs in items:
        rec = Record(str(filename))
        for sa, ra in regs:
            kind, pts = _shape(sa)
            if kind is None:
                if pts:
                    skipped[pts] = skipped.get(pts, 0) + 1
                continue
            cls = ""
            if isinstance(ra, dict):
                cls = _val(ra.get(key)) if key is not None else ""
            rec.objs.append(Obj(kind, cls or "object", pts))
        if not regs and isinstance(fattrs, dict):
            vals = [(k, _val(v)) for k, v in fattrs.items()]
            vals = [x for x in vals if x[1]]
            if len(fattrs) == 1 and len(vals) == 1:
                rec.tag = vals[0][1]
        parsed.records.append(rec)
    for k, n in skipped.items():
        notes.add(f"{fname}: фигуры «{k}» пропущены: {n}")


def _regions_list(r):
    if isinstance(r, dict):
        r = list(r.values())
    out = []
    if isinstance(r, list):
        for x in r:
            if isinstance(x, dict):
                out.append((x.get("shape_attributes"), x.get("region_attributes") or {}))
    return out


def _json_cell(s):
    s = (s or "").strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def _parse_csv(ctx, p, parsed, notes):
    txt = ctx.read_text(p)
    first = txt.lstrip("﻿").split("\n", 1)[0]
    delim = ";" if first.count(";") > first.count(",") else ","
    rd = csv.DictReader(io.StringIO(txt.lstrip("﻿"), newline=""), delimiter=delim)
    groups: dict = {}
    bad = 0
    for row in rd:
        try:
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            g = groups.setdefault((fn, row.get("file_size") or ""), {"regs": [], "fa": _json_cell(row.get("file_attributes"))})
            sa = _json_cell(row.get("region_shape_attributes"))
            if sa:
                g["regs"].append((sa, _json_cell(row.get("region_attributes"))))
        except Exception:  # noqa
            bad += 1
    if bad:
        notes.add(f"{p.name}: битых строк пропущено: {bad}")
    _build([(k[0], g["regs"], g["fa"]) for k, g in groups.items()], None, p.name, parsed, notes)


def parse(ctx, det, opt):
    parsed = Parsed(NAME)
    notes = _Notes(parsed)
    for p in det.files:
        try:
            if p.suffix.lower() == ".csv":
                _parse_csv(ctx, p, parsed, notes)
                continue
            d = ctx.read_json(p)
            ents = _entries(d)
            if not ents:
                notes.add(f"{p.name}: не похоже на VIA, пропущен")
                continue
            via_attrs = d.get("_via_attributes") if isinstance(d, dict) else None
            items = [(e.get("filename"), _regions_list(e.get("regions")), e.get("file_attributes")) for e in ents if e.get("filename")]
            _build(items, via_attrs, p.name, parsed, notes)
        except Exception as e:  # noqa
            notes.add(f"{p.name}: ошибка разбора ({e})")
    if notes.dropped:
        parsed.notes.append(f"... и ещё предупреждений: {notes.dropped}")
    return parsed
