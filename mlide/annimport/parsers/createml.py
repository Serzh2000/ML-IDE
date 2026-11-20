"""Apple CreateML: JSON-список [{"image": ..., "annotations": [{"label", "coordinates": {x, y, width, height}}]}]; x, y - центр."""
from __future__ import annotations


from ..types import Detection, Obj, Parsed, Record

NAME = "createml"
TITLE = {"ru": "Apple CreateML (JSON)", "en": "Apple CreateML (JSON)"}

HEAD = 2_000_000
MAX_SNIFF = 30
MAX_NOTES = 20


def _is_item(t) -> bool:
    if not isinstance(t, dict) or "image" not in t or not isinstance(t.get("annotations"), list) or "data" in t:
        return False
    return all(isinstance(a, dict) and isinstance(a.get("coordinates"), dict) for a in t["annotations"][:3])


def _has_coords(t) -> bool:
    return _is_item(t) and any(isinstance(a, dict) and "coordinates" in a for a in t["annotations"])


def sniff(ctx):
    files, best = [], 0.0
    for i, p in enumerate(ctx.files(".json")):
        if i >= MAX_SNIFF:
            break
        try:
            txt = ctx.read_text(p, limit=HEAD)
        except OSError:
            continue
        if '"coordinates"' not in txt or '"image"' not in txt:
            continue
        d = ctx.read_json(p)
        if not isinstance(d, list) or not d:
            continue
        items = d[:50]
        n = sum(_is_item(t) for t in items)
        if n >= 0.5 * len(items) and any(_has_coords(t) for t in items):
            files.append(p)
            best = max(best, 0.95)
    if not files:
        return None
    return Detection(NAME, best, files, f"{len(files)} файл(ов) CreateML")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse(ctx, det, opt):
    parsed = Parsed(NAME)
    dropped = 0

    def note(msg):
        nonlocal dropped
        if len(parsed.notes) < MAX_NOTES:
            parsed.notes.append(msg)
        else:
            dropped += 1

    for p in det.files:
        try:
            d = ctx.read_json(p)
        except Exception as e:  # noqa
            note(f"{p.name}: ошибка чтения ({e})")
            continue
        if not isinstance(d, list):
            note(f"{p.name}: не похоже на CreateML, пропущен")
            continue
        bad = 0
        recs = []
        for t in d:
            try:
                if not isinstance(t, dict) or not t.get("image"):
                    bad += 1
                    continue
                rec = Record(str(t["image"]))
                for a in t.get("annotations") or []:
                    c = a.get("coordinates") if isinstance(a, dict) else None
                    if not isinstance(c, dict):
                        continue
                    x, y, w, h = (_num(c.get(k)) for k in ("x", "y", "width", "height"))
                    if None in (x, y, w, h) or w <= 0 or h <= 0:
                        continue
                    label = a.get("label")
                    rec.objs.append(Obj("box", "object" if label in (None, "") else str(label),
                                        [[x - w / 2, y - h / 2], [x + w / 2, y + h / 2]]))
                recs.append(rec)
            except Exception:  # noqa
                bad += 1
        boxes = [o for r in recs for o in r.objs]
        if len(boxes) >= 3 and all(o.pts[0][0] < 0 or o.pts[0][1] < 0 for o in boxes):     # x, y - левый верхний угол
            for o in boxes:
                (x1, y1), (x2, y2) = o.pts
                w, h = x2 - x1, y2 - y1
                o.pts = [[x1 + w / 2, y1 + h / 2], [x2 + w / 2, y2 + h / 2]]
            note(f"{p.name}: координаты x, y приняты как левый верхний угол")
        parsed.records.extend(recs)
        if bad:
            note(f"{p.name}: записей с ошибкой пропущено: {bad}")
    if dropped:
        parsed.notes.append(f"... и ещё предупреждений: {dropped}")
    return parsed
