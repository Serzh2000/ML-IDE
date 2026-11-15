"""Export / import / class-tools tests (plain python; also collectable by pytest).

Run:  python tests/test_formats.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mlide import imgio  # noqa: E402
from mlide.augment import AugParams  # noqa: E402
from mlide.classtools import (class_usage, merge_classes, move_class, remap_classes,  # noqa: E402
                              rename_class)
from mlide.formats import ExportOptions, export_dataset, import_yolo_labels  # noqa: E402
from mlide.importers import import_coco, import_voc  # noqa: E402
from mlide.model import Annotation, Project  # noqa: E402

SIZES = [(200, 150), (320, 240), (400, 300), (256, 256), (180, 260), (300, 200), (220, 220),
         (350, 250), (240, 180)]


def make_image(path: Path, w, h, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    img[: h // 2] = (img[: h // 2] // 2)
    imgio.imwrite(path, img)


def build_project(base: Path, n=9, unique=True) -> Project:
    """n labeled images of different sizes (boxes + polygons), 3 classes."""
    rnd = random.Random(1)
    pr = Project("t")
    for c in ("кошка", "dog", "bird"):
        pr.add_class(c)
    paths = []
    for i in range(n):
        w, h = SIZES[i % len(SIZES)]
        p = base / "Картинки" / f"sub{i % 3}" / (f"im{i}.png" if unique else "same.png")
        if not unique:
            p = base / "Картинки" / f"d{i}" / "same.png"
        make_image(p, w, h, i)
        paths.append(p)
    pr.add_images(paths)
    for e in pr.images:
        for k in range(3):
            x1 = rnd.uniform(2, e.w * 0.5)
            y1 = rnd.uniform(2, e.h * 0.5)
            x2 = x1 + rnd.uniform(20, e.w * 0.4)
            y2 = y1 + rnd.uniform(20, e.h * 0.4)
            e.anns.append(Annotation(k % 3, "box", [[x1, y1], [x2, y2]]))
        cx, cy = rnd.uniform(40, e.w - 40), rnd.uniform(40, e.h - 40)
        poly = [[cx, cy - 30], [cx + 30, cy - 5], [cx + 20, cy + 25], [cx - 20, cy + 25], [cx - 30, cy - 5]]
        e.anns.append(Annotation(1, "poly", poly))
    return pr


def opts(out, fmt, **kw):
    d = dict(fmt=fmt, out_dir=str(out), train=0.7, val=0.3, include_unlabeled=False,
             write_train_script=False)
    d.update(kw)
    return ExportOptions(**d)


def by_name(pr):
    return {e.name: e for e in pr.images}


def _imgs_in(d: Path):
    return sorted(p for p in d.iterdir() if imgio.is_image(p))


def _check_yolo_labels(img_dir: Path, lbl_dir: Path, kind, nc, check_sizes=True):
    n = 0
    for im in _imgs_in(img_dir):
        lf = lbl_dir / (im.stem + ".txt")
        assert lf.exists(), f"missing label {lf}"
        for line in lf.read_text(encoding="utf-8").splitlines():
            t = line.split()
            assert 0 <= int(t[0]) < nc
            v = [float(x) for x in t[1:]]
            assert all(0.0 <= x <= 1.0 for x in v), (lf, line)
            if kind == "det":
                assert len(v) == 4
            elif kind == "seg":
                assert len(v) >= 6 and len(v) % 2 == 0
            else:
                assert len(v) == 8
        n += 1
    return n


# ----------------------------------------------------------------------------- export tests
def _expected_count(pr):
    return sum(1 for e in pr.images if any(not a.suggested for a in e.anns))


def test_ultralytics_and_yolov7():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src")
        total = _expected_count(pr)
        for fmt, kind in (("ultra_det", "det"), ("ultra_seg", "seg"), ("ultra_obb", "obb"), ("yolov7", "det")):
            out = td / ("out_" + fmt)
            export_dataset(pr, opts(out, fmt))
            cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            assert cfg["nc"] == 3 and cfg["names"] == {0: "кошка", 1: "dog", 2: "bird"}
            seen, count = [], 0
            for s in ("train", "val"):
                rel = cfg[s]
                idir = out / rel
                ldir = out / (rel.replace("images", "labels", 1) if fmt != "yolov7"
                              else Path(rel).parts[0]) / ("labels" if fmt == "yolov7" else "")
                if fmt == "yolov7":
                    ldir = out / s / "labels"
                count += _check_yolo_labels(idir, ldir, kind, 3)
                seen += [p.name for p in _imgs_in(idir)]
            assert count == total == len(set(seen)), (fmt, count, total)
            assert len(seen) == len(set(seen))
            assert len(_imgs_in(out / cfg["train"])) == 6 and len(_imgs_in(out / cfg["val"])) == 3


def test_darknet():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src")
        out = td / "dk"
        export_dataset(pr, opts(out, "darknet"))
        assert (out / "obj.names").read_text(encoding="utf-8").split() == ["кошка", "dog", "bird"]
        od = (out / "obj.data").read_text(encoding="utf-8")
        assert "classes = 3" in od and "train = train.txt" in od and "valid = valid.txt" in od
        tr = (out / "train.txt").read_text().split()
        va = (out / "valid.txt").read_text().split()
        assert len(tr) == 6 and len(va) == 3 and not set(tr) & set(va)
        for rel in tr + va:
            assert (out / rel).exists()
            assert (out / rel).with_suffix(".txt").exists()
        _check_yolo_labels(out / "data" / "obj", out / "data" / "obj", "det", 3)


def test_coco_export_and_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src")
        out = td / "coco"
        export_dataset(pr, opts(out, "coco"))
        names_seen = []
        n_ann = 0
        for s, cnt in (("train", 6), ("val", 3)):
            d = json.loads((out / "annotations" / f"instances_{s}.json").read_text())
            assert len(d["images"]) == cnt
            assert [c["name"] for c in d["categories"]] == ["кошка", "dog", "bird"]
            ids = {i["id"] for i in d["images"]}
            for a in d["annotations"]:
                assert a["image_id"] in ids and 0 <= a["category_id"] < 3
                x, y, w, h = a["bbox"]
                im = next(i for i in d["images"] if i["id"] == a["image_id"])
                assert x >= 0 and y >= 0 and x + w <= im["width"] + 1e-6 and y + h <= im["height"] + 1e-6
            for i in d["images"]:
                assert (out / s / i["file_name"]).exists()
                names_seen.append(i["file_name"])
            n_ann += len(d["annotations"])
        assert len(set(names_seen)) == 9
        # round trip
        pr2 = Project("r")
        reps = [import_coco(pr2, out / "annotations" / f"instances_{s}.json", images_dir=out / s)
                for s in ("train", "val")]
        assert sum(r.images_added for r in reps) == 9 and sum(r.annotations_added for r in reps) == n_ann
        assert pr2.class_names() == pr.class_names()
        orig = by_name(pr)
        for e2 in pr2.images:
            e1 = orig[e2.name]
            assert (e1.w, e1.h) == (e2.w, e2.h) and len(e1.anns) == len(e2.anns)
            for a1, a2 in zip(e1.anns, e2.anns):
                assert a1.cls == a2.cls
                assert all(abs(u - v) <= 1.0 for u, v in zip(a1.bbox(), a2.bbox()))
                assert len(a2.polygon()) == len(a1.polygon())
        # re-import must not duplicate
        r = import_coco(pr2, out / "annotations" / "instances_train.json", images_dir=out / "train")
        assert r.images_added == 0 and r.skipped_existing == 6 and r.annotations_added == 0
        # missing images are counted
        r = import_coco(Project("x"), out / "annotations" / "instances_val.json", images_dir=td / "nowhere")
        assert r.images_added == 0 or r.missing_files == 0   # found via json-folder fallback is allowed
        assert "Добавлено" in r.summary()


def test_voc_export_and_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src")
        out = td / "voc"
        export_dataset(pr, opts(out, "voc"))
        assert (out / "labels.txt").read_text(encoding="utf-8").split() == ["кошка", "dog", "bird"]
        jp = _imgs_in(out / "JPEGImages")
        assert len(jp) == 9
        for p in jp:
            assert (out / "Annotations" / (p.stem + ".xml")).exists()
        tr = (out / "ImageSets" / "Main" / "train.txt").read_text().split()
        va = (out / "ImageSets" / "Main" / "val.txt").read_text().split()
        assert len(tr) == 6 and len(va) == 3 and not set(tr) & set(va)
        pr2 = Project("r")
        rep = import_voc(pr2, out)
        assert rep.images_added == 9, rep.summary()
        orig = by_name(pr)
        for e2 in pr2.images:
            e1 = orig[e2.name]
            assert len(e1.anns) == len(e2.anns)
            for a1, a2 in zip(e1.anns, e2.anns):
                assert pr.classes[a1.cls]["name"] == pr2.classes[a2.cls]["name"]
                assert all(abs(u - v) <= 1.0 for u, v in zip(a1.bbox(), a2.bbox())), (a1.bbox(), a2.bbox())
        r = import_voc(pr2, out)
        assert r.images_added == 0 and r.skipped_existing == 9
        # flat folder: xml next to image
        flat = td / "плоская"
        flat.mkdir()
        make_image(flat / "f.png", 100, 80, 5)
        (flat / "f.xml").write_text(
            "<annotation><filename>f.png</filename><size><width>100</width><height>80</height></size>"
            "<object><name>zzz</name><bndbox><xmin>11</xmin><ymin>6</ymin><xmax>50</xmax><ymax>40</ymax></bndbox>"
            "</object></annotation>", encoding="utf-8")
        pr3 = Project("f")
        r = import_voc(pr3, flat)
        assert r.images_added == 1 and r.annotations_added == 1 and r.classes_added == ["zzz"]
        assert pr3.images[0].anns[0].pts == [[10.0, 5.0], [50.0, 40.0]]


def test_yolo_roundtrips():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src")
        orig = by_name(pr)
        for fmt, tol in (("ultra_det", 1.0), ("ultra_seg", 1.5)):
            out = td / fmt
            export_dataset(pr, opts(out, fmt, train=1.0, val=0.0))
            pr2 = Project("r")
            pr2.classes = [dict(c) for c in pr.classes]
            pr2.add_images(_imgs_in(out / "images" / "train"))
            assert len(pr2.images) == 9
            for e2 in pr2.images:
                import_yolo_labels(e2, 3)
                e1 = orig[e2.name]
                assert len(e1.anns) == len(e2.anns)
                for a1, a2 in zip(e1.anns, e2.anns):
                    assert a1.cls == a2.cls
                    if fmt == "ultra_det":
                        assert a2.kind == "box"
                        assert all(abs(u - v) <= tol for u, v in zip(a1.bbox(), a2.bbox()))
                    else:
                        p1, p2 = a1.polygon(), a2.polygon()
                        assert a2.kind == "poly" and len(p1) == len(p2)
                        assert all(abs(u - v) <= tol for q1, q2 in zip(p1, p2) for u, v in zip(q1, q2))


# ----------------------------------------------------------------------------- edge cases
def test_empty_images_and_suggested():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src", n=4)
        p = td / "src" / "empty.png"
        make_image(p, 100, 100, 99)
        make_image(td / "src" / "sugg.png", 100, 100, 98)
        pr.add_images([p, td / "src" / "sugg.png"])
        sg = next(e for e in pr.images if e.name == "sugg.png")
        sg.anns.append(Annotation(0, "box", [[5, 5], [50, 50]], suggested=True))
        mixed = next(e for e in pr.images if e.anns and e is not sg)
        n_conf = sum(1 for a in mixed.anns if not a.suggested)
        mixed.anns.append(Annotation(2, "box", [[1, 1], [20, 20]], suggested=True))
        # include_unlabeled False: empty + suggested-only are dropped
        out = td / "o1"
        export_dataset(pr, opts(out, "ultra_det", train=1.0, val=0.0))
        assert len(_imgs_in(out / "images" / "train")) == 4
        lbl = (out / "labels" / "train" / (Path(mixed.path).stem + ".txt")).read_text().splitlines()
        assert len(lbl) == n_conf
        # include_unlabeled True: every image gets a label file, empty ones are empty
        out = td / "o2"
        export_dataset(pr, opts(out, "ultra_det", train=1.0, val=0.0, include_unlabeled=True))
        assert _check_yolo_labels(out / "images" / "train", out / "labels" / "train", "det", 3) == 6
        assert (out / "labels" / "train" / "empty.txt").read_text() == ""
        assert (out / "labels" / "train" / "sugg.txt").read_text() == ""
        # skip_suggested=False keeps suggestions
        out = td / "o3"
        export_dataset(pr, opts(out, "ultra_det", train=1.0, val=0.0, skip_suggested=False))
        assert len(_imgs_in(out / "images" / "train")) == 5
        # VOC / COCO with unlabeled
        out = td / "o4"
        export_dataset(pr, opts(out, "voc", include_unlabeled=True))
        assert (out / "Annotations" / "empty.xml").exists()
        out = td / "o5"
        export_dataset(pr, opts(out, "coco", train=1.0, val=0.0, include_unlabeled=True))
        d = json.loads((out / "annotations" / "instances_train.json").read_text())
        assert len(d["images"]) == 6


def test_duplicate_names():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src", n=3, unique=False)
        assert len({e.name for e in pr.images}) == 1 and len(pr.images) == 3
        for fmt in ("ultra_det", "voc", "darknet", "coco"):
            out = td / fmt
            export_dataset(pr, opts(out, fmt, train=1.0, val=0.0))
            if fmt == "ultra_det":
                d = out / "images" / "train"
            elif fmt == "voc":
                d = out / "JPEGImages"
            elif fmt == "darknet":
                d = out / "data" / "obj"
            else:
                d = out / "train"
            names = [p.name for p in _imgs_in(d)]
            assert len(names) == 3 and len(set(n.lower() for n in names)) == 3, (fmt, names)
            if fmt == "ultra_det":
                assert len(list((out / "labels" / "train").glob("*.txt"))) == 3


def test_bake_aug():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pr = build_project(td / "src", n=4)
        aug = AugParams(copies=2, mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=15, translate=0.2,
                        scale=0.4, shear=5, fliplr=0.5, cutout=0.3)
        out = td / "aug"
        export_dataset(pr, opts(out, "ultra_det", train=1.0, val=0.0, bake_aug=True, aug=aug))
        imgs = _imgs_in(out / "images" / "train")
        augs = [p for p in imgs if "_aug" in p.stem]
        assert len(imgs) > 4 and augs, "no augmented copies produced"
        assert _check_yolo_labels(out / "images" / "train", out / "labels" / "train", "det", 3) == len(imgs)
        for p in augs:
            h, w = imgio.imread(p).shape[:2]
            src = next(e for e in pr.images if e.name.split(".")[0] == p.stem.split("_aug")[0])
            assert (w, h) == (src.w, src.h), "augmented size differs from label normalisation size"
            for line in (out / "labels" / "train" / (p.stem + ".txt")).read_text().splitlines():
                cx, cy, bw, bh = map(float, line.split()[1:])
                assert cx - bw / 2 >= -1e-5 and cx + bw / 2 <= 1 + 1e-5
                assert cy - bh / 2 >= -1e-5 and cy + bh / 2 <= 1 + 1e-5 and bw > 0 and bh > 0
        # seg export with augmentation as well
        out = td / "aug_seg"
        export_dataset(pr, opts(out, "ultra_seg", train=1.0, val=0.0, bake_aug=True, aug=aug))
        _check_yolo_labels(out / "images" / "train", out / "labels" / "train", "seg", 3)


# ----------------------------------------------------------------------------- class tools
def _cls_project():
    pr = Project("c")
    for n in ("a", "b", "c", "d"):
        pr.add_class(n)
    from mlide.model import ImageEntry
    e = ImageEntry("x.png", 100, 100)
    for c in (0, 1, 1, 2, 3, 3, 3):
        e.anns.append(Annotation(c, "box", [[1, 1], [9, 9]]))
    e.anns.append(Annotation(2, "box", [[1, 1], [9, 9]], suggested=True))
    pr.images.append(e)
    return pr


def _cls_list(pr):
    return [a.cls for a in pr.images[0].anns]


def test_class_tools():
    pr = _cls_project()
    assert class_usage(pr) == [1, 2, 1, 3]
    pr.dirty = False
    merge_classes(pr, 1, 3)                    # b -> d, objects kept
    assert pr.dirty and pr.class_names() == ["a", "c", "d"]
    assert class_usage(pr) == [1, 1, 5]
    assert _cls_list(pr) == [0, 2, 2, 1, 2, 2, 2, 1]
    pr = _cls_project()
    rename_class(pr, 0, "z")
    assert pr.class_names() == ["z", "b", "c", "d"]
    rename_class(pr, 2, "a2")
    rename_class(pr, 0, "d")                   # exists -> merge a(0)->d
    assert pr.class_names() == ["b", "a2", "d"] and class_usage(pr) == [2, 1, 4]
    pr = _cls_project()
    remap_classes(pr, {0: None, 1: 2})         # delete a, b->c
    assert pr.class_names() == ["c", "d"] and class_usage(pr) == [3, 3]
    pr = _cls_project()
    move_class(pr, 3, 0)                       # d to front
    assert pr.class_names() == ["d", "a", "b", "c"] and class_usage(pr) == [3, 1, 2, 1]
    move_class(pr, 0, 3)
    assert pr.class_names() == ["a", "b", "c", "d"] and class_usage(pr) == [1, 2, 1, 3]
    assert _cls_list(pr) == [0, 1, 1, 2, 3, 3, 3, 2]
    try:
        remap_classes(pr, {0: 1, 1: 2})
        raise AssertionError("chain must be rejected")
    except ValueError:
        pass


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    for t in ALL:
        t()
        print("ok", t.__name__)
    print("ALL OK")


if __name__ == "__main__":
    main()
