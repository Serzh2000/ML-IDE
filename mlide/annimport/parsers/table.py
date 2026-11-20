"""Таблицы: CSV / TSV / JSON-списки / JSON Lines, где каждая строка связывает картинку с классом и/или рамкой.

Столбцы определяются по названиям (filename, class, xmin, ymin, xmax, ymax, bbox, polygon, width, height...) и по содержимому;
сопоставление можно поправить в окне импорта: оно лежит в det.params["mapping"] и в opt.params["mapping"].
Формат координат: xyxy (углы), xywh (левый верхний угол + размеры), cxcywh (центр + размеры), строка bbox, строка polygon.
Единицы: пиксели или доли кадра (определяются сами, можно задать unit = "px" | "norm").
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from ... import imgio
from ..types import Detection, Obj, Parsed, Record

NAME = "table"
TITLE = {"ru": "Таблица (CSV / TSV / JSON)", "en": "Table (CSV / TSV / JSON)"}

IMAGE_NAMES = ["filename", "file_name", "filepath", "file_path", "file", "image", "images", "imagepath", "image_path", "imagefile",
               "image_file", "imagename", "image_name", "img", "img_path", "imgpath", "imgname", "path", "fname", "frame", "url", "uri",
               "picture", "photo", "imageid", "image_id", "id"]
CLASS_NAMES = ["class", "classname", "class_name", "label", "labelname", "label_name", "category", "categoryname", "category_name",
               "object", "objectname", "cls", "type", "target", "species", "tag", "labels", "class_id", "classid", "category_id", "name"]
POLY_NAMES = ["polygon", "points", "segmentation", "poly", "contour", "polygons", "all_points", "mask_points"]
BBOX_STR_NAMES = ["bbox", "box", "bounding_box", "boundingbox", "bndbox", "bboxes", "boxes", "rect", "rectangle", "region"]
W_NAMES = ["width", "imagewidth", "image_width", "img_width", "imgwidth", "imagew", "image_w", "img_w", "orig_width", "original_width"]
H_NAMES = ["height", "imageheight", "image_height", "img_height", "imgheight", "imageh", "image_h", "img_h", "orig_height", "original_height"]

# наборы столбцов рамки: (формат, [имена по порядку])
BOX_SETS = [
    ("xyxy", ["xmin", "ymin", "xmax", "ymax"]), ("xyxy", ["x_min", "y_min", "x_max", "y_max"]), ("xyxy", ["x1", "y1", "x2", "y2"]),
    ("xyxy", ["left", "top", "right", "bottom"]), ("xyxy", ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]),
    ("xyxy", ["x_left", "y_top", "x_right", "y_bottom"]), ("xyxy", ["minx", "miny", "maxx", "maxy"]),
    ("xyxy", ["xtl", "ytl", "xbr", "ybr"]), ("xyxy", ["tlx", "tly", "brx", "bry"]),
    ("cxcywh", ["xc", "yc", "w", "h"]), ("cxcywh", ["cx", "cy", "w", "h"]), ("cxcywh", ["cx", "cy", "width", "height"]),
    ("cxcywh", ["xcenter", "ycenter", "width", "height"]), ("cxcywh", ["x_center", "y_center", "width", "height"]),
    ("cxcywh", ["xcenter", "ycenter", "w", "h"]), ("cxcywh", ["center_x", "center_y", "width", "height"]),
    ("xywh", ["x", "y", "w", "h"]), ("xywh", ["x", "y", "width", "height"]), ("xywh", ["left", "top", "width", "height"]),
    ("xywh", ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]), ("xywh", ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]),
    ("xywh", ["xmin", "ymin", "width", "height"]), ("xywh", ["x_min", "y_min", "width", "height"]),
]
OBJ_KEYS = ["objects", "annotations", "boxes", "bboxes", "regions", "labels", "shapes", "detections", "items", "instances", "bounding_boxes"]
ROW_KEYS = ["data", "rows", "records", "images", "items", "annotations", "results", "samples"]


def _n(s) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(s).strip().lower().replace(" ", "_").replace("-", "_"))


def _is_img(v) -> bool:
    return isinstance(v, str) and imgio.is_image(v.strip().split("?")[0])


def _num(v):
    try:
        return float(str(v).strip().replace(",", "."))
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- загрузка строк
def _delim(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max(",;\t|", key=lambda d: first.count(d))


def _flat_json_rows(items: list) -> list[dict]:
    """Список элементов JSON -> плоские строки (вложенные объекты разворачиваются в отдельные строки)."""
    rows: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        base = {}
        nested = None
        for k, v in it.items():
            if _n(k) in OBJ_KEYS and isinstance(v, (list, dict)) and nested is None and v:
                nested = v
            elif isinstance(v, dict) and all(not isinstance(x, (list, dict)) for x in v.values()):
                base.update({f"{k}.{kk}": vv for kk, vv in v.items()})
                base[k] = v
            else:
                base[k] = v
        if nested is None:
            rows.append(base)
            continue
        objs = []
        if isinstance(nested, dict):                                   # HF: {"bbox": [...], "category": [...]}
            keys = [k for k, v in nested.items() if isinstance(v, list)]
            n = max((len(nested[k]) for k in keys), default=0)
            for i in range(n):
                objs.append({k: nested[k][i] for k in keys if i < len(nested[k])})
        else:
            objs = [o for o in nested if isinstance(o, dict)] or [{"label": o} for o in nested if isinstance(o, str)]
        if not objs:
            rows.append(base)
        for o in objs:
            r = dict(base)
            for k, v in o.items():
                if isinstance(v, dict) and all(not isinstance(x, (list, dict)) for x in v.values()):
                    for kk, vv in v.items():
                        r[kk] = vv
                else:
                    r[k] = v
            rows.append(r)
    return rows


def load_rows(ctx, f: Path, limit: int | None = None):
    """-> (header: list[str], rows: list[dict], headerless: bool). Для CSV значения - строки, для JSON - как есть."""
    ext = f.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        items = ctx.read_jsonl(f, max_lines=limit)
        rows = _flat_json_rows(items)
        return list(dict.fromkeys(k for r in rows[:200] for k in r)), rows, False
    if ext == ".json":
        obj = ctx.read_json(f)
        items = None
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if _n(k) in ROW_KEYS and isinstance(v, list) and v and isinstance(v[0], dict):
                    items = v
                    break
            if items is None and obj and all(isinstance(v, dict) for v in obj.values()):
                items = [dict(v, **{"_key": k}) for k, v in obj.items()]
        if not items:
            return [], [], False
        rows = _flat_json_rows(items[:limit] if limit else items)
        return list(dict.fromkeys(k for r in rows[:200] for k in r)), rows, False
    text = ctx.read_text(f, limit=(None if limit is None else 3_000_000))
    if not text.strip():
        return [], [], False
    delim = ";" if ext == ".list" else _delim(text[:20000])
    rd = list(csv.reader(io.StringIO(text), delimiter=delim, skipinitialspace=True))
    rd = [r for r in rd if any(c.strip() for c in r)]
    if limit:
        rd = rd[: limit + 1]
    if not rd:
        return [], [], False
    first = rd[0]
    # заголовок есть, если первая строка содержит известные названия столбцов или нет ни одного числа/пути картинки
    known = {_n(c) for c in first} & set(IMAGE_NAMES + CLASS_NAMES + W_NAMES + H_NAMES + BBOX_STR_NAMES + POLY_NAMES +
                                         [c for _fmt, cs in BOX_SETS for c in cs])
    has_img = any(_is_img(c) for c in first)
    has_num = any(_num(c) is not None for c in first)
    headerless = (has_img or has_num) and not known
    if headerless:
        header = [f"col{i + 1}" for i in range(len(first))]
        body = rd
    else:
        header = [c.strip() or f"col{i + 1}" for i, c in enumerate(first)]
        body = rd[1:]
    rows = [{h: (r[i].strip() if i < len(r) else "") for i, h in enumerate(header)} for r in body]
    return header, rows, headerless


# --------------------------------------------------------------------------- определение столбцов
def _col(header, names, exclude=()):
    low = {_n(h): h for h in header if h not in exclude}
    for nm in names:
        if nm in low:
            return low[nm]
    return None


def infer_mapping(header: list[str], rows: list[dict]) -> dict | None:
    """Угадывает роли столбцов по названиям и содержимому. None - если картинку найти не удалось."""
    sample = rows[:200]
    m: dict = {"image": None, "class": None, "box": None, "poly": None, "width": None, "height": None, "unit": "auto"}
    # картинка: столбец, где большинство значений похожи на имя файла картинки
    best, best_frac = None, 0.0
    for h in header:
        vals = [r.get(h) for r in sample if r.get(h) not in (None, "")]
        if vals:
            fr = sum(_is_img(v) for v in vals) / len(vals)
            if fr > best_frac:
                best, best_frac = h, fr
    if best and best_frac >= 0.6:
        m["image"] = best
    else:
        m["image"] = _col(header, IMAGE_NAMES)
    if not m["image"]:
        return None
    used = {m["image"]}
    # рамка по набору столбцов
    low = {_n(h): h for h in header}
    for fmt, names in BOX_SETS:
        if all(n in low for n in names):
            m["box"] = {"format": fmt, "cols": [low[n] for n in names]}
            used |= set(m["box"]["cols"])
            break
    if m["box"] is None:                                                # рамка в одном столбце: "[x, y, w, h]" / "x y w h"
        c = _col(header, BBOX_STR_NAMES, used)
        if c:
            vals = [r.get(c) for r in sample if r.get(c) not in (None, "")]
            ok = 0
            for v in vals[:50]:
                nums = _nums_in(v)
                ok += len(nums) == 4
            if vals and ok >= 0.8 * min(len(vals), 50):
                m["box"] = {"format": "str", "col": c, "order": "xywh"}
                used.add(c)
    if m["box"] is None:
        c = _col(header, POLY_NAMES, used)
        if c:
            vals = [r.get(c) for r in sample if r.get(c) not in (None, "")]
            if vals and sum(len(_nums_in(v)) >= 6 for v in vals[:50]) >= 0.8 * min(len(vals), 50):
                m["poly"] = {"col": c}
                used.add(c)
    # размеры кадра: width/height считаются размером кадра, если рамка задана другими столбцами
    wc = _col(header, W_NAMES, used)
    hc = _col(header, H_NAMES, used)
    if wc and hc:
        m["width"], m["height"] = wc, hc
        used |= {wc, hc}
    # класс
    cc = _col(header, CLASS_NAMES, used)
    if cc:
        m["class"] = cc
    elif m["box"] is None and m["poly"] is None:
        # таблица «картинка + метка» без названий: берём первый нечисловой столбец, не являющийся картинкой
        for h in header:
            if h in used:
                continue
            vals = [r.get(h) for r in sample if r.get(h) not in (None, "")]
            if vals and sum(_num(v) is None for v in vals) >= 0.8 * len(vals) and len(set(map(str, vals))) <= max(2, len(vals) // 2):
                m["class"] = h
                break
        if m["class"] is None:
            for h in header:                                            # числовая метка (class_id)
                if h not in used:
                    vals = [r.get(h) for r in sample if r.get(h) not in (None, "")]
                    if vals and all(_num(v) is not None and float(_num(v)).is_integer() for v in vals):
                        m["class"] = h
                        break
    if m["box"] is None and m["poly"] is None and m["class"] is None:
        return None
    # headerless: четыре числа подряд после столбца картинки - рамка xyxy (формат Keras-RetinaNet: path,x1,y1,x2,y2,class)
    if m["box"] is None and m["poly"] is None and all(re.fullmatch(r"col\d+", h) for h in header):
        idx = header.index(m["image"])
        cols = header[idx + 1: idx + 5]
        if len(cols) == 4 and all(sum(_num(r.get(c)) is not None for r in sample) >= 0.8 * max(1, len(sample)) for c in cols):
            m["box"] = {"format": "xyxy", "cols": cols}
            rest = [h for h in header if h not in set(cols) | {m["image"]}]
            m["class"] = rest[0] if rest else None
    return m


def _nums_in(v) -> list[float]:
    if isinstance(v, (list, tuple)):
        flat = []
        for x in v:
            flat += _nums_in(x)
        return flat
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(v))]


# --------------------------------------------------------------------------- разбор
def _box_pts(fmt, v):
    a, b, c, d = v
    if fmt == "xyxy":
        return [[a, b], [c, d]]
    if fmt == "xywh":
        return [[a, b], [a + c, b + d]]
    return [[a - c / 2, b - d / 2], [a + c / 2, b + d / 2]]          # cxcywh


def build_parsed(rows: list[dict], mapping: dict, name: str = "table", unit_hint: str | None = None) -> Parsed:
    pd = Parsed(name)
    unit = unit_hint or mapping.get("unit", "auto")
    # единицы: доли кадра, если все координаты в 0..1 (и размеров кадра нет или они больше 1)
    if unit == "auto":
        mx, seen = 0.0, False
        for r in rows[:2000]:
            vals = _coords(r, mapping)
            if vals:
                seen = True
                mx = max(mx, max(abs(x) for x in vals))
        unit = "norm" if seen and mx <= 1.0001 else "px"
    order: dict[str, Record] = {}
    img_c, cls_c = mapping["image"], mapping.get("class")
    wc, hc = mapping.get("width"), mapping.get("height")
    for r in rows:
        ref = r.get(img_c)
        if ref in (None, ""):
            continue
        ref = str(ref).strip()
        rec = order.get(ref)
        if rec is None:
            rec = order[ref] = Record(ref=ref)
            if wc and hc:
                w, h = _num(r.get(wc)), _num(r.get(hc))
                if w and h and w > 1 and h > 1:
                    rec.width, rec.height = int(w), int(h)
        cls = r.get(cls_c) if cls_c else None
        cls = None if cls in (None, "") else (str(cls).strip() if not isinstance(cls, (int, float)) else
                                              (str(int(cls)) if float(cls).is_integer() else str(cls)))
        pts = kind = None
        box = mapping.get("box")
        if box:
            if box["format"] == "str":
                v = _nums_in(r.get(box["col"]))
                if len(v) == 4:
                    pts, kind = _box_pts(box.get("order", "xywh"), v), "box"
            else:
                v = [_num(r.get(c)) for c in box["cols"]]
                if all(x is not None for x in v):
                    pts, kind = _box_pts(box["format"], v), "box"
        elif mapping.get("poly"):
            v = _nums_in(r.get(mapping["poly"]["col"]))
            if len(v) >= 6 and len(v) % 2 == 0:
                pts, kind = [[v[i], v[i + 1]] for i in range(0, len(v), 2)], "poly"
        if pts is not None:
            if cls is None:
                continue
            rec.objs.append(Obj(kind, cls, pts, norm=(unit == "norm")))
        elif cls is not None and not box and not mapping.get("poly"):
            rec.tag = cls
        elif cls is not None and rec.tag is None and not rec.objs:
            pass                                                         # строка-картинка без координат: ничего не добавляем
    pd.records = list(order.values())
    return pd


def _coords(r: dict, mapping: dict) -> list[float]:
    box = mapping.get("box")
    if box:
        v = _nums_in(r.get(box["col"])) if box["format"] == "str" else [_num(r.get(c)) for c in box["cols"]]
        return [x for x in v if x is not None]
    if mapping.get("poly"):
        return _nums_in(r.get(mapping["poly"]["col"]))
    return []


def table_files(ctx) -> list[Path]:
    return ctx.files(".csv", ".tsv", ".json", ".jsonl", ".ndjson")


def sniff(ctx):
    best = None
    for f in table_files(ctx)[:60]:
        try:
            if f.stat().st_size < 10 or f.stat().st_size > 800_000_000:
                continue
            header, rows, _hl = load_rows(ctx, f, limit=300)
        except Exception:  # noqa
            continue
        if not rows or len(header) < 2:
            continue
        m = infer_mapping(header, rows)
        if not m:
            continue
        has_geo = bool(m["box"] or m["poly"])
        score = (0.8 if has_geo else 0.62) if f.suffix.lower() in (".csv", ".tsv") else (0.5 if has_geo else 0.42)
        if best is None or score > best.score:
            kind = "рамки" if m["box"] else "полигоны" if m["poly"] else "классы кадров"
            best = Detection(NAME, score, [f], f"{f.name}: {len(rows)}+ строк, {kind}", {"mapping": m, "file": str(f)})
    return best


def parse(ctx, det, opt) -> Parsed:
    f = det.files[0]
    header, rows, _hl = load_rows(ctx, f)
    mapping = (opt.params or {}).get("mapping") or det.params.get("mapping") or infer_mapping(header, rows)
    if not mapping:
        return Parsed("table", notes=["Не удалось определить столбцы таблицы: укажите их вручную."])
    pd = build_parsed(rows, mapping, "table", (opt.params or {}).get("unit"))
    if not pd.records:
        pd.notes.append("В таблице нет строк с картинками.")
    return pd
