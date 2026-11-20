"""LabelMe / labelImg-json: по одному JSON на картинку (shapes + imagePath)."""
from __future__ import annotations

from ..types import Detection, Obj, Parsed, Record

NAME = "labelme"
TITLE = {"ru": "LabelMe (JSON)", "en": "LabelMe (JSON)"}

HEAD = 2_000_000
MAX_SNIFF = 30
MAX_NOTES = 20


def _is_lm(d) -> bool:
    return isinstance(d, dict) and isinstance(d.get("shapes"), list) and "imagePath" in d


def sniff(ctx):
    allf = ctx.files(".json")
    hits = seen = 0
    for p in allf[:MAX_SNIFF]:
        try:
            txt = ctx.read_text(p, limit=HEAD)
        except OSError:
            continue
        seen += 1
        if '"shapes"' in txt and '"imagePath"' in txt and _is_lm(ctx.read_json(p)):
            hits += 1
    if not hits:
        return None
    score = 0.95 if hits >= 0.5 * seen else 0.8
    return Detection(NAME, score, list(allf), f"{len(allf)} файл(ов) JSON, похоже на LabelMe")


def _pts(sh):
    out = []
    for q in sh.get("points") or []:
        try:
            out.append([float(q[0]), float(q[1])])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
    return out


def parse(ctx, det, opt):
    parsed = Parsed(NAME)
    notes = parsed.notes
    dropped = 0
    skipped_kinds: dict = {}

    def note(msg):
        nonlocal dropped
        if len(notes) < MAX_NOTES:
            notes.append(msg)
        else:
            dropped += 1

    for p in det.files:
        try:
            d = ctx.read_json(p)
            if not _is_lm(d):
                continue
            ip = d.get("imagePath")
            rec = Record(str(ip) if ip else p.stem)
            rec.extra_refs.append(p.stem)
            try:
                rec.width = int(d.get("imageWidth")) or None
            except (TypeError, ValueError):
                pass
            try:
                rec.height = int(d.get("imageHeight")) or None
            except (TypeError, ValueError):
                pass
            for sh in d.get("shapes") or []:
                if not isinstance(sh, dict):
                    continue
                st = str(sh.get("shape_type") or "polygon").lower()
                label = sh.get("label")
                label = "object" if label in (None, "") else str(label)
                pts = _pts(sh)
                if pts is None:
                    note(f"{p.name}: некорректные точки фигуры")
                    continue
                if st == "rectangle" and len(pts) >= 2:
                    rec.objs.append(Obj("box", label, [pts[0], pts[1]]))
                elif st == "polygon" and len(pts) >= 3:
                    rec.objs.append(Obj("poly", label, pts))
                elif st == "circle" and len(pts) >= 2:
                    (cx, cy), (ex, ey) = pts[0], pts[1]
                    r = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
                    rec.objs.append(Obj("box", label, [[cx - r, cy - r], [cx + r, cy + r]]))
                elif st not in ("rectangle", "polygon", "circle"):
                    skipped_kinds[st] = skipped_kinds.get(st, 0) + 1
            flags = d.get("flags")
            if isinstance(flags, dict):
                on = [k for k, v in flags.items() if v is True]
                if len(on) == 1:
                    rec.tag = str(on[0])
            parsed.records.append(rec)
        except Exception as e:  # noqa
            note(f"{p.name}: ошибка разбора ({e})")
    names = {"point": "точки", "points": "точки", "line": "линии", "linestrip": "ломаные", "mask": "маски"}
    for k, n in skipped_kinds.items():
        note(f"Фигуры «{names.get(k, k)}» пропущены: {n}")
    if dropped:
        notes.append(f"... и ещё предупреждений: {dropped}")
    return parsed
