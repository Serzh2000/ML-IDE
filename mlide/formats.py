"""Dataset export (YOLO variants, COCO, VOC) and YOLO label import."""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
import yaml

from . import imgio
from .augment import AugParams, augment_sample
from .model import Annotation, ImageEntry, Project

FORMATS = {
    "ultra_det": "YOLO Детекция - Ultralytics (v5 / v8 / v9 / v10 / v11 / v12 / 26)",
    "ultra_seg": "YOLO Сегментация - Ultralytics (модели -seg)",
    "ultra_obb": "YOLO Повёрнутые рамки - Ultralytics (модели -obb)",
    "yolov7": "YOLOv7 / YOLOR / структура Roboflow (train/images, train/labels)",
    "darknet": "Darknet YOLOv3 / v4 / Scaled-YOLOv4 (obj.names, obj.data, train.txt)",
    "coco": "COCO JSON (YOLOX, RT-DETR, Detectron2, MMDetection...)",
    "voc": "Pascal VOC XML",
}

TASK_OF = {"ultra_det": "detect", "ultra_seg": "segment", "ultra_obb": "obb"}

MODEL_FAMILIES = {
    "YOLO26": ("yolo26{s}", "yolo26{s}-seg", "yolo26{s}-obb"),
    "YOLO12": ("yolo12{s}", "yolo12{s}-seg", "yolo12{s}-obb"),
    "YOLO11": ("yolo11{s}", "yolo11{s}-seg", "yolo11{s}-obb"),
    "YOLOv10": ("yolov10{s}", None, None),
    "YOLOv9": ("yolov9{s}", "yolov9{s}-seg", None),
    "YOLOv8": ("yolov8{s}", "yolov8{s}-seg", "yolov8{s}-obb"),
    "YOLOv5 (u)": ("yolov5{s}u", "yolov5{s}-seg", None),
}


@dataclass
class ExportOptions:
    fmt: str = "ultra_det"
    out_dir: str = ""
    train: float = 0.8
    val: float = 0.2
    test: float = 0.0
    seed: int = 0
    include_unlabeled: bool = False      # keep background images (empty label files)
    only_reviewed: bool = False
    skip_suggested: bool = True          # ignore un-accepted auto-suggestions
    bake_aug: bool = False               # offline augmentation of the train split
    aug: AugParams = field(default_factory=AugParams)
    write_train_script: bool = True
    family: str = "YOLO11"
    size: str = "n"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    online_aug: bool = True
    freeze: int = 0            # заморозка первых N слоёв (дообучение на малых данных)
    lr0: float = 0.0           # 0 = по умолчанию; >0 = AdamW с этим шагом
    patience: int = 100
    group_split: bool = False      # похожие кадры (дубли, кадры одного видео) - только в одну часть выборки
    group_hash: bool = True        # искать визуально похожие картинки
    group_names: bool = True       # группировать по шаблону имени файла (clip_0001, clip_0002…)


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------------------- geometry helpers
def _items(entry: ImageEntry, skip_suggested):
    return [a for a in entry.anns if not (skip_suggested and a.suggested)]


def _obb_points(a: Annotation):
    pts = np.asarray(a.polygon(), dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect)


def _clip_box(a: Annotation, w, h):
    """Bounding box clipped to the image, or None when less than 1 px of it is inside.
    Out-of-range coordinates make Ultralytics reject the whole image as corrupt."""
    x1, y1, x2, y2 = a.bbox()
    x1, x2 = min(max(x1, 0), w), min(max(x2, 0), w)
    y1, y2 = min(max(y1, 0), h), min(max(y2, 0), h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def _owned_dirs(out: Path, fmt):
    """Folders a format fully owns inside the export folder; emptied before every export so that files of
    a previous export (another split / seed / filter) cannot leak into the new dataset."""
    if fmt in ("ultra_det", "ultra_seg", "ultra_obb"):
        return [out / "images", out / "labels"]
    if fmt == "yolov7":
        return [out / s / d for s in ("train", "val", "test") for d in ("images", "labels")]
    if fmt == "darknet":
        return [out / "data" / "obj"]
    if fmt == "coco":
        return [out / "train", out / "val", out / "test", out / "annotations"]
    if fmt == "voc":
        return [out / "JPEGImages", out / "Annotations", out / "ImageSets"]
    raise ValueError(fmt)


def _clean_previous(project: Project, out: Path, fmt):
    owned = [d.resolve() for d in _owned_dirs(out, fmt)]
    for e in project.images:
        src = Path(e.path).resolve()
        if any(d in src.parents for d in owned):
            raise ValueError("Папка экспорта содержит исходные изображения проекта "
                             f"({src.parent}). Выберите для экспорта другую папку.")
    for d in owned:
        if d.is_dir():
            shutil.rmtree(d)
    if fmt == "darknet":
        for n in ("train.txt", "valid.txt", "test.txt"):
            (out / n).unlink(missing_ok=True)


def yolo_lines(fmt, anns, w, h):
    out = []
    for a in anns:
        if fmt in ("ultra_det", "yolov7", "darknet"):
            box = _clip_box(a, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            out.append(f"{a.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        elif fmt == "ultra_seg":
            pts = a.polygon()
            flat = " ".join(f"{min(max(x / w, 0), 1):.6f} {min(max(y / h, 0), 1):.6f}" for x, y in pts)
            out.append(f"{a.cls} {flat}")
        elif fmt == "ultra_obb":
            pts = _obb_points(a)
            flat = " ".join(f"{min(max(x / w, 0), 1):.6f} {min(max(y / h, 0), 1):.6f}" for x, y in pts)
            out.append(f"{a.cls} {flat}")
    return "\n".join(out) + ("\n" if out else "")


# ----------------------------------------------------------------------------- export
def _split(items, o: ExportOptions, progress=None):
    if getattr(o, "group_split", False) and len(items) > 1:
        from .dedupe import build_groups, split_groups
        groups = build_groups([e.path for e in items], use_hash=o.group_hash, use_names=o.group_names,
                              progress=(lambda d, t: progress(d, t, "поиск похожих кадров")) if progress else None)
        res = split_groups(items, lambda e: groups[e.path], {"train": o.train, "val": o.val, "test": o.test}, o.seed)
        return {k: v for k, v in res.items() if v}
    rnd = random.Random(o.seed)
    items = list(items)
    rnd.shuffle(items)
    tot = max(1e-9, o.train + o.val + o.test)
    n = len(items)
    n_train = int(round(n * o.train / tot))
    n_val = int(round(n * o.val / tot))
    if o.val > 0 and n >= 2 and n_val == 0:
        n_val = 1
    n_train = min(n_train, n - n_val)
    splits = {"train": items[:n_train], "val": items[n_train:n_train + n_val],
              "test": items[n_train + n_val:]}
    return {k: v for k, v in splits.items() if v}


def _unique(name, used):
    """Unique by stem, not by full name: cat.jpg and cat.png would otherwise share one cat.txt label file."""
    stem, ext = Path(name).stem, Path(name).suffix
    cand, i = stem, 1
    while cand.lower() in used:
        cand = f"{stem}_{i}"
        i += 1
    used.add(cand.lower())
    return cand + ext


def _copy_image(src: str, dst: Path):
    """Copy as-is unless EXIF rotation would make labels disagree with pixels."""
    if imgio.exif_orientation(src) in (2, 3, 4, 5, 6, 7, 8):
        imgio.imwrite(dst, imgio.imread(src))
    else:
        shutil.copyfile(src, dst)


def export_dataset(project: Project, o: ExportOptions, progress=lambda i, n, m: True):
    """Returns a short report string."""
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = project.class_names()
    if not names:
        raise ValueError("В проекте нет классов.")

    entries = []
    for e in project.images:
        if o.only_reviewed and not e.reviewed:
            continue
        if not _items(e, o.skip_suggested) and not o.include_unlabeled:
            continue
        entries.append(e)
    if not entries:
        raise ValueError("Нечего экспортировать: нет размеченных изображений, подходящих под фильтры.")

    _clean_previous(project, out, o.fmt)
    splits = _split(entries, o, progress)
    total = sum(len(v) for v in splits.values())
    done = 0
    rng = np.random.default_rng(o.seed)
    used: dict[str, set] = {}       # per image folder: VOC and Darknet put every split into one folder
    stats = {"images": 0, "labels": 0, "aug": 0}
    coco = {s: {"images": [], "annotations": [], "categories": [
        {"id": i, "name": n} for i, n in enumerate(names)]} for s in splits}
    coco_ann_id = 1
    darknet_lists = {s: [] for s in splits}

    def dirs(split):
        if o.fmt in ("ultra_det", "ultra_seg", "ultra_obb"):
            return out / "images" / split, out / "labels" / split
        if o.fmt == "yolov7":
            return out / split / "images", out / split / "labels"
        if o.fmt == "darknet":
            return out / "data" / "obj", out / "data" / "obj"
        if o.fmt == "coco":
            return out / split, None
        if o.fmt == "voc":
            return out / "JPEGImages", out / "Annotations"
        raise ValueError(o.fmt)

    def emit(split, entry, img_name, anns_triplets, image_src=None, image_arr=None, w=None, h=None):
        nonlocal coco_ann_id
        idir, ldir = dirs(split)
        idir.mkdir(parents=True, exist_ok=True)
        if ldir:
            ldir.mkdir(parents=True, exist_ok=True)
        if image_arr is not None:
            imgio.imwrite(idir / img_name, image_arr)
        else:
            _copy_image(image_src, idir / img_name)
        anns = [Annotation(c, k, [list(p) for p in pts]) for c, k, pts in anns_triplets]
        stem = Path(img_name).stem
        stats["images"] += 1
        stats["labels"] += len(anns)
        if o.fmt in ("ultra_det", "ultra_seg", "ultra_obb", "yolov7", "darknet"):
            (ldir / f"{stem}.txt").write_text(yolo_lines(o.fmt, anns, w, h), encoding="utf-8")
            if o.fmt == "darknet":
                darknet_lists[split].append(f"data/obj/{img_name}")
        elif o.fmt == "coco":
            iid = len(coco[split]["images"]) + 1
            coco[split]["images"].append({"id": iid, "file_name": img_name, "width": w, "height": h})
            for a in anns:
                box = _clip_box(a, w, h)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                poly = [[min(max(float(x), 0), w), min(max(float(y), 0), h)] for x, y in a.polygon()]
                area = (x2 - x1) * (y2 - y1) if a.kind == "box" else cv2.contourArea(np.float32(poly))
                coco[split]["annotations"].append({
                    "id": coco_ann_id, "image_id": iid, "category_id": a.cls,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "area": round(float(area), 2), "iscrowd": 0,
                    "segmentation": [[round(v, 2) for p in poly for v in p]]})
                coco_ann_id += 1
        elif o.fmt == "voc":
            objs = []
            for a in anns:
                box = _clip_box(a, w, h)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                xmin, ymin = min(int(round(x1)) + 1, w), min(int(round(y1)) + 1, h)     # VOC is 1-based, inclusive
                xmax, ymax = max(xmin, int(round(x2))), max(ymin, int(round(y2)))
                objs.append(f"<object><name>{escape(names[a.cls])}</name><pose>Unspecified</pose>"
                            f"<truncated>0</truncated><difficult>0</difficult><bndbox>"
                            f"<xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
                            f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>")
            xml = (f"<annotation><folder>JPEGImages</folder><filename>{escape(img_name)}</filename>"
                   f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"
                   f"<segmented>0</segmented>{''.join(objs)}</annotation>")
            (ldir / f"{stem}.xml").write_text(xml, encoding="utf-8")
            (out / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
            with open(out / "ImageSets" / "Main" / f"{split}.txt", "a", encoding="utf-8") as f:
                f.write(stem + "\n")

    for split, items in splits.items():
        for e in items:
            if not progress(done, total, e.name):
                raise Cancelled()
            done += 1
            anns = [(a.cls, a.kind, a.pts) for a in _items(e, o.skip_suggested)]
            names_used = used.setdefault(str(dirs(split)[0]), set())
            name = _unique(e.name, names_used)
            emit(split, e, name, anns, image_src=e.path, w=e.w, h=e.h)
            if o.bake_aug and split == "train" and o.aug.copies > 0 and anns:
                img = imgio.imread(e.path)
                for k in range(o.aug.copies):
                    aimg, aanns = augment_sample(img, anns, o.aug, rng)
                    if not aanns:
                        continue
                    aname = _unique(f"{Path(name).stem}_aug{k}.jpg", names_used)
                    emit(split, e, aname, aanns, image_arr=aimg, w=e.w, h=e.h)
                    stats["aug"] += 1

    # ---- dataset descriptor files
    if o.fmt in ("ultra_det", "ultra_seg", "ultra_obb", "yolov7"):
        # no absolute "path": it is resolved relative to data.yaml, so the folder can be moved / uploaded
        cfg = {}
        for s in splits:
            cfg[s] = f"images/{s}" if o.fmt != "yolov7" else f"{s}/images"
        cfg["nc"] = len(names)
        cfg["names"] = {i: n for i, n in enumerate(names)}
        (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    elif o.fmt == "darknet":
        (out / "obj.names").write_text("\n".join(names) + "\n", encoding="utf-8")
        for s, lst in darknet_lists.items():
            (out / f"{'valid' if s == 'val' else s}.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")
        valid = "valid.txt" if "val" in splits else "train.txt"
        (out / "obj.data").write_text(
            f"classes = {len(names)}\ntrain = train.txt\nvalid = {valid}\nnames = obj.names\nbackup = backup/\n",
            encoding="utf-8")
        (out / "backup").mkdir(exist_ok=True)
    elif o.fmt == "coco":
        (out / "annotations").mkdir(exist_ok=True)
        for s, d in coco.items():
            (out / "annotations" / f"instances_{s}.json").write_text(json.dumps(d), encoding="utf-8")
    elif o.fmt == "voc":
        (out / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    if o.write_train_script and o.fmt in TASK_OF:
        write_train_script(out, o)

    sp = ", ".join(f"{k}: {len(v)}" for k, v in splits.items())
    return (f"Экспортировано изображений: {stats['images']} (из них аугментированных: {stats['aug']}), "
            f"объектов: {stats['labels']}.\nРазделение - {sp}\n-> {out}")


# ----------------------------------------------------------------------------- training script
def write_train_script(out: Path, o: ExportOptions):
    task = TASK_OF[o.fmt]
    idx = {"detect": 0, "segment": 1, "obb": 2}[task]
    tmpl = MODEL_FAMILIES.get(o.family, MODEL_FAMILIES["YOLO11"])[idx]
    if tmpl is None:
        tmpl = MODEL_FAMILIES[o.family][0]
        note = f"# NOTE: {o.family} has no official '{task}' weights - falling back to detection weights.\n"
    else:
        note = ""
    weights = tmpl.format(s=o.size) + ".pt"
    hyp = o.aug.ultralytics_hyp()
    if not o.online_aug:            # omitting the keys would silently enable the trainer's own defaults
        hyp = {k: 0.0 for k in hyp}
    aug_note = "augmentation preset (online, applied on the fly by the trainer)" if o.online_aug \
        else "online augmentation switched off in ML IDE"
    hyp_lines = "\n".join(f"        {k}={v!r}," for k, v in hyp.items())
    extra = [f"        patience={o.patience},"]
    if o.freeze:
        extra.append(f"        freeze={o.freeze},")
    if o.lr0 > 0:
        extra.append('        optimizer="AdamW",')
        extra.append(f"        lr0={o.lr0},")
    extra_lines = "\n".join(extra)
    code = f'''"""Auto-generated by ML IDE.  Run:  pip install ultralytics  &&  python train.py"""
from pathlib import Path
from ultralytics import YOLO

{note}HERE = Path(__file__).resolve().parent


def main():
    model = YOLO("{weights}")   # pretrained weights are downloaded automatically
    model.train(
        data=str(HERE / "data.yaml"),
        epochs={o.epochs},
        imgsz={o.imgsz},
        batch={o.batch},
        project=str(HERE / "runs"),
{extra_lines}
        # {aug_note}
{hyp_lines}
    )
    model.val()
    model.export(format="onnx")   # ONNX can be loaded back into ML IDE (Tools > Auto-annotate)


if __name__ == "__main__":      # required on Windows: data-loader workers re-import this file
    main()
'''
    (out / "train.py").write_text(code, encoding="utf-8")


# ----------------------------------------------------------------------------- YOLO label import
def read_class_names(path):
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        n = d.get("names", [])
        return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def find_label_file(img_path: str):
    p = Path(img_path)
    cands = [p.with_suffix(".txt")]
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            q = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            cands.insert(0, q)
            break
    return next((c for c in cands if c.exists()), None)


def import_yolo_labels(entry: ImageEntry, n_classes: int) -> int:
    lf = find_label_file(entry.path)
    if not lf:
        return 0
    n = 0
    for line in lf.read_text(encoding="utf-8").splitlines():
        t = line.split()
        try:
            cls, v = int(float(t[0])), [float(x) for x in t[1:]]
        except (ValueError, IndexError):
            continue
        if cls >= n_classes:
            continue
        if len(v) == 4:
            cx, cy, bw, bh = v
            entry.anns.append(Annotation(cls, "box", [[(cx - bw / 2) * entry.w, (cy - bh / 2) * entry.h],
                                                     [(cx + bw / 2) * entry.w, (cy + bh / 2) * entry.h]]))
        elif len(v) >= 6 and len(v) % 2 == 0:
            pts = [[v[i] * entry.w, v[i + 1] * entry.h] for i in range(0, len(v), 2)]
            entry.anns.append(Annotation(cls, "poly", pts))
        else:
            continue
        n += 1
    return n
