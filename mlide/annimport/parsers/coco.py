"""COCO: instances / detection / keypoints JSON (images + annotations + categories)."""
from __future__ import annotations

from ..types import Detection, Obj, Parsed, Record

NAME = "coco"
TITLE = {"ru": "COCO (JSON)", "en": "COCO (JSON)"}

HEAD = 2_000_000
MAX_SNIFF = 30
MAX_NOTES = 20


def _is_coco(d) -> int:
    """0 - не COCO; 1 - слабо (images + annotations/categories); 2 - уверенно (images + categories + annotations)."""
    if not isinstance(d, dict) or not isinstance(d.get("images"), list):
        return 0
    has_c = isinstance(d.get("categories"), list)
    has_a = isinstance(d.get("annotations"), list)
    if has_c and has_a:
        return 2
    if has_c or has_a:
        return 1
    return 0


def _head_ok(txt: str) -> bool:
    return '"images"' in txt and ('"categories"' in txt or '"annotations"' in txt or len(txt) >= HEAD)


def _is_results_head(txt: str) -> bool:
    return txt.lstrip()[:1] == "[" and '"image_id"' in txt and '"category_id"' in txt


def sniff(ctx):
    files, strong, nres = [], 0, []
    for i, p in enumerate(ctx.files(".json")):
        if i >= MAX_SNIFF:
            break
        try:
            txt = ctx.read_text(p, limit=HEAD)
        except OSError:
            continue
        if _head_ok(txt):
            lvl = _is_coco(ctx.read_json(p))
            if lvl:
                files.append(p)
                strong = max(strong, lvl)
        elif _is_results_head(txt):
            nres.append(p)
    if not files:
        return None
    files = files + nres
    score = 0.95 if strong == 2 else 0.7
    return Detection(NAME, score, files, f"{len(files)} файл(ов) COCO")


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _poly_parts(seg):
    out = []
    if isinstance(seg, list):
        for s in seg:
            if isinstance(s, list) and len(s) >= 6:
                nums = [_num(v) for v in s]
                if any(v is None for v in nums):
                    continue
                out.append([[nums[j], nums[j + 1]] for j in range(0, len(nums) - 1, 2)])
    return out


def _bbox(bb):
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        x, y, w, h = [_num(v) for v in bb]
        if None in (x, y, w, h):
            return None
        return [[x, y], [x + w, y + h]]
    return None


def _kps_triples(flat):
    out = []
    if isinstance(flat, list):
        for j in range(0, len(flat) - 2, 3):
            x, y, v = _num(flat[j], 0.0), _num(flat[j + 1], 0.0), _num(flat[j + 2], 0)
            out.append([x, y, int(v)])
    return out


def parse(ctx, det, opt):
    parsed = Parsed(NAME)
    notes = parsed.notes
    dropped = [0]

    def note(msg):
        if len(notes) < MAX_NOTES:
            notes.append(msg)
        else:
            dropped[0] += 1

    img_index: dict = {}      # id -> [(папка, Record)] для файлов-результатов
    cat_global: dict = {}
    results = []
    coco_files = []
    for p in det.files:
        try:
            d = ctx.read_json(p)
        except Exception:  # noqa
            d = None
        if isinstance(d, list):
            results.append((p, d))
        elif _is_coco(d):
            coco_files.append((p, d))
        else:
            note(f"{p.name}: не похоже на COCO, пропущен")

    for p, d in coco_files:
        try:
            cats = {}
            for c in d.get("categories") or []:
                if isinstance(c, dict) and "id" in c:
                    cats[str(c["id"])] = str(c.get("name", c["id"]))
                    if c.get("keypoints") and not parsed.keypoints:
                        parsed.keypoints = [str(k) for k in c["keypoints"]]
                        sk = []
                        for pr in c.get("skeleton") or []:
                            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                                try:
                                    sk.append([int(pr[0]), int(pr[1])])
                                except (TypeError, ValueError):
                                    pass
                        if sk and min(min(a) for a in sk) >= 1:
                            sk = [[a - 1, b - 1] for a, b in sk]
                        parsed.skeleton = sk
            for k, v in cats.items():
                cat_global.setdefault(k, v)
            recs = {}
            order = []
            for im in d.get("images") or []:
                if not isinstance(im, dict) or not im.get("file_name"):
                    continue
                rec = Record(str(im["file_name"]))
                w, h = _num(im.get("width")), _num(im.get("height"))
                rec.width = int(w) if w else None
                rec.height = int(h) if h else None
                for key in ("coco_url", "flickr_url", "path", "file_path", "filename", "image_path", "url"):
                    v = im.get(key)
                    if isinstance(v, str) and v.strip() and v != rec.ref and v not in rec.extra_refs:
                        rec.extra_refs.append(v)
                if "id" in im:
                    rec.extra_refs.append(str(im["id"]))
                    recs[str(im["id"])] = rec
                    img_index.setdefault(str(im["id"]), []).append((p.parent, rec))
                order.append(rec)
            parsed.records.extend(order)
            skipped = 0
            for a in d.get("annotations") or []:
                if not isinstance(a, dict):
                    continue
                rec = recs.get(str(a.get("image_id")))
                if rec is None:
                    continue
                if a.get("iscrowd") or isinstance(a.get("segmentation"), dict):
                    skipped += 1
                    continue
                cid = str(a.get("category_id"))
                _add_ann(rec, a, cats.get(cid, cid))
            if skipped:
                note(f"{p.name}: RLE/crowd-маски пропущены ({skipped})")
        except Exception as e:  # noqa
            note(f"{p.name}: ошибка разбора ({e})")

    for p, lst in results:
        if not img_index:
            note(f"{p.name}: нет COCO-файла с картинками рядом, результаты пропущены")
            continue
        n = 0
        for a in lst:
            if not isinstance(a, dict) or "image_id" not in a:
                continue
            cands = img_index.get(str(a["image_id"]))
            if not cands:
                continue
            rec = next((r for dd, r in cands if dd == p.parent), cands[0][1])
            cid = str(a.get("category_id"))
            _add_ann(rec, a, cat_global.get(cid, cid), conf=_num(a.get("score")))
            n += 1
        if not n:
            note(f"{p.name}: ни один результат не совпал с картинками")
    if dropped[0]:
        notes.append(f"... и ещё предупреждений: {dropped[0]}")
    return parsed


def _is_axis_rect(pl, tol: float = 0.5) -> bool:
    """Четыре угла осевого прямоугольника (в любом порядке обхода; замыкающая точка допускается)."""
    pts = [list(p) for p in pl]
    if len(pts) == 5 and abs(pts[0][0] - pts[4][0]) <= tol and abs(pts[0][1] - pts[4][1]) <= tol:
        pts = pts[:4]
    if len(pts) != 4:
        return False
    xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
    return abs(xs[0] - xs[1]) <= tol and abs(xs[2] - xs[3]) <= tol and abs(ys[0] - ys[1]) <= tol and abs(ys[2] - ys[3]) <= tol         and xs[3] - xs[0] > tol and ys[3] - ys[0] > tol


def _add_ann(rec, a, cls, conf=None):
    kps = _kps_triples(a.get("keypoints"))
    box = _bbox(a.get("bbox"))
    if kps and any(k[2] > 0 for k in kps):
        if box is None or box[1][0] - box[0][0] <= 0 or box[1][1] - box[0][1] <= 0:
            vis = [k for k in kps if k[2] > 0]
            xs, ys = [k[0] for k in vis], [k[1] for k in vis]
            box = [[min(xs), min(ys)], [max(xs), max(ys)]]
        rec.objs.append(Obj("pose", cls, box, kps=kps, conf=conf))
        return
    polys = _poly_parts(a.get("segmentation"))
    if len(polys) == 1 and _is_axis_rect(polys[0]):          # экспорт рамок в COCO часто пишет прямоугольник как полигон: это рамка
        xs, ys = [p[0] for p in polys[0]], [p[1] for p in polys[0]]
        rec.objs.append(Obj("box", cls, [[min(xs), min(ys)], [max(xs), max(ys)]], conf=conf))
        return
    if polys:
        for pl in polys:
            rec.objs.append(Obj("poly", cls, pl, conf=conf))
        return
    if box is not None:
        rec.objs.append(Obj("box", cls, box, conf=conf))
