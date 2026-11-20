"""Label Studio: полный JSON-экспорт (задачи data/annotations/predictions), JSON-MIN и JSONL."""
from __future__ import annotations

import json
import math
import re
import urllib.parse

from ... import imgio
from ..types import Detection, Obj, Parsed, Record

NAME = "labelstudio"
TITLE = {"ru": "Label Studio (JSON)", "en": "Label Studio (JSON)"}

HEAD = 2_000_000
MAX_SNIFF = 30
MAX_NOTES = 20
LABEL_KEYS = ("rectanglelabels", "polygonlabels", "keypointlabels", "brushlabels", "ellipselabels", "labels")


def _is_full(t) -> bool:
    return isinstance(t, dict) and isinstance(t.get("data"), dict) and (
        isinstance(t.get("annotations"), list) or isinstance(t.get("predictions"), list))


def _min_results(t):
    """Плоская запись JSON-MIN -> [(ключ, список словарей с метками)]."""
    out = []
    if not isinstance(t, dict) or "data" in t:
        return out
    for k, v in t.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v) and any(
                key in x for x in v for key in LABEL_KEYS):
            out.append((k, v))
    return out


def _is_min(t) -> bool:
    return bool(_min_results(t))


def _head_items(ctx, p, txt):
    """Первые несколько задач файла для проверки формата (без полного разбора больших файлов)."""
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        items = []
        for ln in txt.splitlines()[:5]:
            try:
                items.append(json.loads(ln))
            except ValueError:
                continue
        return items
    d = ctx.read_json(p)
    if isinstance(d, dict):
        d = [d]
    return d[:50] if isinstance(d, list) else []


def sniff(ctx):
    files, best = [], 0.0
    for i, p in enumerate(ctx.files(".json", ".jsonl", ".ndjson")):
        if i >= MAX_SNIFF:
            break
        try:
            txt = ctx.read_text(p, limit=HEAD)
        except OSError:
            continue
        if not (('"data"' in txt and ('"annotations"' in txt or '"predictions"' in txt)) or "labels\"" in txt):
            continue
        items = _head_items(ctx, p, txt)
        if not items:
            continue
        full = sum(_is_full(t) for t in items)
        mn = sum(_is_min(t) for t in items)
        if full >= 0.5 * len(items):
            files.append(p)
            best = max(best, 0.95)
        elif mn >= 0.5 * len(items) and any(isinstance(t, dict) and any(isinstance(v, str) for v in t.values()) for t in items):
            files.append(p)
            best = max(best, 0.85)
    if not files:
        return None
    return Detection(NAME, best, files, f"{len(files)} файл(ов) Label Studio")


# ---------------------------------------------------------------- ссылки на картинки
_UP = re.compile(r"^/?data/upload/\d+/")
_HEX = re.compile(r"^[0-9a-fA-F]{8}-(?=.+\.[A-Za-z0-9]{2,5}$)")


def _ref_variants(raw: str):
    """-> (основная ссылка, [запасные])"""
    s = raw.strip()
    orig = s
    if "?d=" in s and "data/local-files" in s:
        s = urllib.parse.unquote(s.split("?d=", 1)[1].split("&", 1)[0])
    elif s.startswith("/data/") or s.startswith("data/"):
        s = urllib.parse.unquote(s.split("?", 1)[0])
    s = s.replace("\\", "/")
    s = _UP.sub("", s)
    base, sep, name = s.rpartition("/")
    name2 = _HEX.sub("", name)
    main = base + sep + name2
    extras = []
    for v in (s, orig):
        if v and v != main and v not in extras:
            extras.append(v)
    return main, extras


def _ref_of(data: dict):
    cand = data.get("image")
    if not (isinstance(cand, str) and cand.strip()):
        cand = None
        for v in data.values():
            if isinstance(v, str) and v.strip() and imgio.is_image(v.split("?", 1)[0]):
                cand = v
                break
    if cand is None:
        for k in ("ocr", "url"):
            if isinstance(data.get(k), str) and data[k].strip():
                cand = data[k]
                break
    return cand


# ---------------------------------------------------------------- геометрия
def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _rect(val, ow, oh):
    x, y, w, h = _f(val.get("x")), _f(val.get("y")), _f(val.get("width")), _f(val.get("height"))
    rot = _f(val.get("rotation"))
    if abs(rot) < 1e-6 or w <= 0 or h <= 0:
        return "box", [[x / 100, y / 100], [(x + w) / 100, (y + h) / 100]]
    sx, sy = (ow or 100.0), (oh or 100.0)           # проценты -> пиксели (для верного поворота при неквадратном кадре)
    px, py, pw, ph = x / 100 * sx, y / 100 * sy, w / 100 * sx, h / 100 * sy
    a = math.radians(rot)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for dx, dy in ((0, 0), (pw, 0), (pw, ph), (0, ph)):
        pts.append([(px + dx * c - dy * s) / sx, (py + dx * s + dy * c) / sy])
    return "obb", pts


def _first_label(val, key):
    lab = val.get(key)
    if isinstance(lab, list) and lab:
        return str(lab[0])
    if isinstance(lab, str) and lab:
        return lab
    return None


class _Ctx:
    def __init__(self, parsed):
        self.parsed, self.dropped, self.kinds = parsed, 0, {}

    def note(self, msg):
        if len(self.parsed.notes) < MAX_NOTES:
            self.parsed.notes.append(msg)
        else:
            self.dropped += 1

    def skip(self, what):
        self.kinds[what] = self.kinds.get(what, 0) + 1


def _apply_results(rec, results, nc, kp_names):
    """results: [{"id","type","parent_id","value","original_width","original_height"}]"""
    boxes_by_id = {}
    kp_groups: dict = {}
    pending = []
    for r in results:
        if not isinstance(r, dict) or not isinstance(r.get("value"), dict):
            continue
        t = str(r.get("type") or "")
        val = r["value"]
        ow, oh = r.get("original_width") or val.get("original_width"), r.get("original_height") or val.get("original_height")
        try:
            if ow and oh and not rec.width:
                rec.width, rec.height = int(ow), int(oh)
        except (TypeError, ValueError):
            pass
        ow, oh = _f(ow) or None, _f(oh) or None
        if t == "choices":
            ch = val.get("choices")
            if isinstance(ch, list) and ch and rec.tag is None:
                rec.tag = str(ch[0])
        elif t == "rectanglelabels":
            cls = _first_label(val, t)
            if cls is None:
                continue
            kind, pts = _rect(val, ow, oh)
            o = Obj(kind, cls, pts, norm=True)
            rec.objs.append(o)
            if r.get("id") is not None:
                boxes_by_id[r["id"]] = o
        elif t == "polygonlabels":
            cls = _first_label(val, t)
            pts = [[_f(p[0]) / 100, _f(p[1]) / 100] for p in val.get("points") or [] if isinstance(p, (list, tuple)) and len(p) >= 2]
            if cls is not None and len(pts) >= 3:
                rec.objs.append(Obj("poly", cls, pts, norm=True))
        elif t == "keypointlabels":
            pid = r.get("parent_id")
            lab = _first_label(val, t) or ""
            if pid is not None:
                kp_groups.setdefault(pid, []).append((lab, _f(val.get("x")) / 100, _f(val.get("y")) / 100))
            else:
                pending.append(lab)
        elif t in ("brushlabels", "brush"):
            nc.skip("маски-кисть")
        elif t in ("rectangle", "polygon", "labels", "textarea", "taxonomy", "number", "rating", "relation", "ellipselabels", "ellipse"):
            if t in ("rectangle", "polygon", "ellipselabels", "ellipse"):
                nc.skip(t)
    for pid, kps in kp_groups.items():
        if len(kps) < 2:
            nc.skip("одиночные ключевые точки")
            continue
        parent = boxes_by_id.get(pid)
        for lab, _x, _y in kps:
            if lab and lab not in kp_names:
                kp_names.append(lab)
        if kp_names and all(lab in kp_names for lab, _x, _y in kps):
            arr = [[0.0, 0.0, 0] for _ in kp_names]
            for lab, x, y in kps:
                arr[kp_names.index(lab)] = [x, y, 2]
        else:
            arr = [[x, y, 2] for _l, x, y in kps]
        xs, ys = [k[0] for k in arr if k[2]], [k[1] for k in arr if k[2]]
        if parent is not None and parent.kind == "box":
            rec.objs.remove(parent)
            rec.objs.append(Obj("pose", parent.cls, parent.pts, kps=arr, norm=True))
        else:
            rec.objs.append(Obj("pose", "object", [[min(xs), min(ys)], [max(xs), max(ys)]], kps=arr, norm=True))
    if pending:
        nc.skip("одиночные ключевые точки")


def _tasks(ctx, p):
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        return ctx.read_jsonl(p)
    d = ctx.read_json(p)
    if isinstance(d, dict):
        return [d]
    return d if isinstance(d, list) else []


def _pick_result(t):
    for a in t.get("annotations") or []:
        if isinstance(a, dict) and not a.get("was_cancelled") and isinstance(a.get("result"), list):
            return a["result"]
    for a in t.get("predictions") or []:
        if isinstance(a, dict) and isinstance(a.get("result"), list):
            return a["result"]
    return []


def _min_to_results(t):
    res = []
    for _k, lst in _min_results(t):
        for x in lst:
            for key in LABEL_KEYS:
                if key in x:
                    res.append({"id": x.get("id"), "parent_id": x.get("parent_id"), "type": key, "value": x,
                                "original_width": x.get("original_width"), "original_height": x.get("original_height")})
                    break
    ch = t.get("choice")
    if isinstance(ch, dict):
        ch = ch.get("choices")
    if isinstance(ch, str) and ch.strip():
        ch = [ch]
    if isinstance(ch, list) and ch:
        res.append({"type": "choices", "value": {"choices": [str(c) for c in ch]}})
    return res


def parse(ctx, det, opt):
    parsed = Parsed(NAME)
    nc = _Ctx(parsed)
    kp_names: list = []
    for p in det.files:
        try:
            tasks = _tasks(ctx, p)
        except Exception as e:  # noqa
            nc.note(f"{p.name}: ошибка чтения ({e})")
            continue
        bad = 0
        for t in tasks:
            try:
                if not isinstance(t, dict):
                    continue
                if _is_full(t):
                    raw = _ref_of(t["data"])
                    results = _pick_result(t)
                elif _is_min(t):
                    raw = _ref_of(t)
                    results = _min_to_results(t)
                else:
                    continue
                if not raw:
                    bad += 1
                    continue
                main, extras = _ref_variants(str(raw))
                rec = Record(main, extra_refs=extras)
                _apply_results(rec, results, nc, kp_names)
                parsed.records.append(rec)
            except Exception:  # noqa
                bad += 1
        if bad:
            nc.note(f"{p.name}: задач без ссылки на картинку или с ошибкой: {bad}")
    if kp_names:
        parsed.keypoints = kp_names
        for rec in parsed.records:
            for o in rec.objs:
                if o.kind == "pose" and o.kps is not None and len(o.kps) < len(kp_names):
                    o.kps = o.kps + [[0.0, 0.0, 0] for _ in range(len(kp_names) - len(o.kps))]
    for k, n in nc.kinds.items():
        nc.note(f"Пропущено ({k}): {n}")
    if nc.dropped:
        parsed.notes.append(f"... и ещё предупреждений: {nc.dropped}")
    return parsed
