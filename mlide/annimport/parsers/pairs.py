"""Файл-сопоставление «картинка -> файл с описанием»: таблица, где один столбец - картинка, другой - путь к файлу разметки
(txt / xml / json). Сами файлы разметки читаются обычными разборщиками (YOLO, VOC, LabelMe...) - формат определяется по первому из них.

Примеры: pairs.csv со строками «img/001.jpg,labels/001.txt»; JSON {"img/001.jpg": "ann/001.xml"}; список из двух столбцов в .txt/.list.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from ... import imgio
from ..context import SourceContext
from ..types import Detection, Options, Parsed

NAME = "pairs"
TITLE = {"ru": "Список «картинка -> файл разметки»", "en": "List \"image -> annotation file\""}
ANN_SUFFIX = {".txt", ".xml", ".json", ".jsonl"}
SUB_PARSERS = ["coco", "voc", "labelme", "cvat", "via", "labelstudio", "createml", "txtper", "yolo"]


def _pair_rows(ctx, f: Path):
    """-> [(ссылка_на_картинку, ссылка_на_файл_разметки)] из таблицы, JSON-словаря или текстового списка."""
    from .table import load_rows
    out = []
    if f.suffix.lower() == ".json":
        obj = ctx.read_json(f)
        if isinstance(obj, dict) and obj and all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
            return list(obj.items())
    if f.suffix.lower() in (".txt", ".list"):                               # «путь_картинки путь_разметки» через пробел/таб
        for ln in ctx.read_text(f).splitlines():
            parts = re.split(r"\t|,|;|\s{2,}|\s(?=\S+\.(?:txt|xml|json)\s*$)", ln.strip())
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) == 2:
                out.append((parts[0], parts[1]))
        return out
    header, rows, _hl = load_rows(ctx, f)
    if len(header) < 2:
        return []
    img_c = ann_c = None
    for h in header:
        vals = [r.get(h) for r in rows[:100] if r.get(h) not in (None, "")]
        if not vals:
            continue
        if sum(imgio.is_image(str(v)) for v in vals) >= 0.8 * len(vals):
            img_c = img_c or h
        elif sum(Path(str(v)).suffix.lower() in ANN_SUFFIX for v in vals) >= 0.8 * len(vals):
            ann_c = ann_c or h
    if not (img_c and ann_c):
        return []
    return [(str(r[img_c]).strip(), str(r[ann_c]).strip()) for r in rows if r.get(img_c) and r.get(ann_c)]


def _resolve_ann(ctx, table: Path, ref: str, by_name: dict) -> Path | None:
    s = ref.replace("\\", "/")
    for base in (table.parent, *ctx.image_roots()):
        for cand in (Path(base) / s, Path(s)):
            if cand.is_file():
                return cand
    return by_name.get(os.path.basename(s).lower())


def sniff(ctx):
    for f in ctx.files(".csv", ".tsv", ".json", ".txt", ".list")[:80]:
        try:
            if f.stat().st_size > 200_000_000:
                continue
            pr = _pair_rows(ctx, f)
        except Exception:  # noqa
            continue
        if len(pr) >= 1:
            by_name = {p.name.lower(): p for p in ctx.files(*ANN_SUFFIX)}
            ok = sum(_resolve_ann(ctx, f, a, by_name) is not None for _i, a in pr[:50])
            if ok >= 0.7 * min(len(pr), 50):
                return Detection(NAME, 0.97, [f], f"{f.name}: {len(pr)} пар «картинка -> файл разметки»", {"file": str(f)})
    return None


def _sub_parser(sub_ctx):
    from .. import detect
    for d in detect(sub_ctx):
        if d.parser in SUB_PARSERS:
            return d
    return None


def parse(ctx, det, opt) -> Parsed:
    import mlide.annimport as ai
    f = det.files[0]
    pairs = _pair_rows(ctx, f)
    by_name = {p.name.lower(): p for p in ctx.files(*ANN_SUFFIX)}
    pd = Parsed("pairs")
    module = None
    first_det = None
    missing = 0
    names_ctx = None
    for img_ref, ann_ref in pairs:
        ap = _resolve_ann(ctx, f, ann_ref, by_name)
        if ap is None:
            missing += 1
            continue
        sub = SourceContext([ap])
        if module is None:
            first_det = _sub_parser(sub)
            if first_det is None:
                pd.notes.append(f"Формат файла разметки не распознан: {ap.name}")
                continue
            module = ai.get_parsers()[first_det.parser]
        d = module.sniff(sub) or first_det.__class__(module.NAME, 0.5, [ap])
        try:
            sp = module.parse(sub, d, opt if isinstance(opt, Options) else Options())
        except Exception as e:  # noqa
            pd.notes.append(f"{ap.name}: {type(e).__name__}")
            continue
        if sp.class_names and not pd.class_names:
            pd.class_names = sp.class_names
        if sp.keypoints and not pd.keypoints:
            pd.keypoints, pd.skeleton = sp.keypoints, sp.skeleton
        if len(sp.records) == 1:                                    # ссылка на картинку - из списка, а не из самого файла
            rec = sp.records[0]
            rec.extra_refs = [rec.ref, *rec.extra_refs]
            rec.ref = img_ref
            pd.records.append(rec)
        else:
            for rec in sp.records:
                pd.records.append(rec)
    if first_det and first_det.parser == "yolo" and not pd.class_names:        # data.yaml / classes.txt лежат рядом, а не в sub-ctx
        from . import yolo
        nm, _k, _y = yolo._read_cfg(ctx)
        pd.class_names = nm or yolo._names_from_txt(ctx, f.parent)
    if missing:
        pd.notes.append(f"Файлов разметки не найдено: {missing}")
    if module is not None:
        pd.format = f"pairs/{module.NAME}"
    return pd
