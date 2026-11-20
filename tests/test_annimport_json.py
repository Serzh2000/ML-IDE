"""Импорт разметки: COCO, LabelMe, VIA, Label Studio, CreateML (разбор, узнавание формата, запись в проект).
Запуск: python tests/test_annimport_json.py"""
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mlide import imgio
from mlide.annimport import Options, SourceContext, detect, parse, run_import
from mlide.model import Project

W, H = 200, 100


def mk(tmp: Path, names, sub=""):
    d = tmp / sub if sub else tmp
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        imgio.imwrite(d / n, np.full((H, W, 3), 50 + i * 20, np.uint8))
    return d


def jdump(p: Path, obj, bom=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8-sig" if bom else "utf-8")
    return p


def top(tmp, name):
    ctx = SourceContext([tmp])
    dets = detect(ctx)
    assert dets and dets[0].parser == name, [(d.parser, d.score) for d in dets]
    return ctx, dets[0]


def approx(a, b, eps=1e-6):
    return abs(a - b) < eps


def run(tmp, ctx, det):
    pr = Project("t")
    rep = run_import(pr, ctx, det, Options())
    return pr, rep


# ------------------------------------------------------------------ COCO
def test_coco():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["a.jpg", "b.jpg", "c.jpg"], "img")
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": W, "height": H},
                   {"id": 2, "file_name": "b.jpg", "width": W, "height": H},
                   {"id": 3, "file_name": "c.jpg", "width": W, "height": H}],
        "categories": [{"id": 7, "name": "cat"}, {"id": 9, "name": "dog"},
                       {"id": 11, "name": "person", "keypoints": ["nose", "eye"], "skeleton": [[1, 2]]}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 20, 30, 40], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 9, "bbox": [0, 0, 10, 10],
             "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10], [50, 50, 60, 50, 60, 60]]},
            {"id": 3, "image_id": 2, "category_id": 7, "bbox": [1, 1, 5, 5], "iscrowd": 1,
             "segmentation": {"counts": "abc", "size": [H, W]}},
            {"id": 4, "image_id": 2, "category_id": 11, "bbox": [10, 10, 50, 50], "keypoints": [20, 20, 2, 0, 0, 0]},
        ],
    }
    jdump(tmp / "ann" / "instances_train.json", coco, bom=True)
    jdump(tmp / "ann" / "other.json", {"hello": "world"})
    jdump(tmp / "ann" / "broken.json", {"images": 5})
    (tmp / "ann" / "bad.json").write_text("{not json", encoding="utf-8")
    jdump(tmp / "ann" / "results.json", [{"image_id": 3, "category_id": 9, "bbox": [5, 5, 20, 20], "score": 0.8}])
    ctx, det = top(tmp, "coco")
    assert det.score >= 0.9
    ps = parse(ctx, det, Options())
    recs = {r.ref.split("/")[-1]: r for r in ps.records}          # имя файла может получить подсказку папки (instances_train.json -> train/)
    assert set(recs) == {"a.jpg", "b.jpg", "c.jpg"}
    a = recs["a.jpg"]
    assert [o.kind for o in a.objs] == ["box", "poly", "poly"] and a.objs[0].cls == "cat"
    assert a.objs[0].pts == [[10, 20], [40, 60]] and a.width == W
    assert "1" not in a.extra_refs          # id картинки не используется как имя файла
    b = recs["b.jpg"]
    assert [o.kind for o in b.objs] == ["pose"] and b.objs[0].kps[0] == [20, 20, 2]
    assert ps.keypoints == ["nose", "eye"] and ps.skeleton == [[0, 1]]
    assert any("RLE" in n for n in ps.notes)
    c = recs["c.jpg"]
    assert len(c.objs) == 1 and c.objs[0].cls == "dog" and c.objs[0].conf == 0.8
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 3 and rep.objects_added == 5 and rep.unmatched == [], rep
    assert pr.keypoints == ["nose", "eye"]


def test_coco_empty_image_and_unrelated():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["x.png"])
    jdump(tmp / "i.json", {"images": [{"id": 1, "file_name": "x.png"}], "annotations": [], "categories": [{"id": 1, "name": "q"}]})
    ctx, det = top(tmp, "coco")
    ps = parse(ctx, det, Options())
    assert len(ps.records) == 1 and ps.records[0].objs == []
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 1


# ------------------------------------------------------------------ LabelMe
def test_labelme():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["p1.jpg", "p2.jpg"])
    jdump(tmp / "p1.json", {"version": "5", "flags": {"good": True, "bad": False}, "imagePath": "p1.jpg",
                            "imageHeight": H, "imageWidth": W, "imageData": None,
                            "shapes": [
                                {"label": "cat", "points": [[10, 10], [50, 40]], "shape_type": "rectangle"},
                                {"label": "dog", "points": [[0, 0], [20, 0], [20, 20]], "shape_type": "polygon"},
                                {"label": "ball", "points": [[100, 50], [110, 50]], "shape_type": "circle"},
                                {"label": "pt", "points": [[1, 1]], "shape_type": "point"},
                                {"label": "ln", "points": [[1, 1], [5, 5]], "shape_type": "linestrip"}]})
    jdump(tmp / "p2.json", {"imagePath": "..\\p2.jpg", "imageWidth": W, "imageHeight": H, "flags": {},
                            "shapes": [{"label": "cat", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"}]}, bom=True)
    (tmp / "broken.json").write_text("{oops", encoding="utf-8")
    jdump(tmp / "noise.json", {"a": 1})
    ctx, det = top(tmp, "labelme")
    ps = parse(ctx, det, Options())
    assert len(ps.records) == 2
    r1 = next(r for r in ps.records if r.ref == "p1.jpg")
    assert [o.kind for o in r1.objs] == ["box", "poly", "box"] and r1.tag == "good" and r1.width == W
    assert r1.objs[2].pts == [[90, 40], [110, 60]]
    assert any("точки" in n for n in ps.notes) and any("ломаные" in n for n in ps.notes)
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 2 and rep.objects_added == 4 and rep.tags_set == 1, rep


# ------------------------------------------------------------------ VIA
def test_via_json():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["v1.jpg", "v2.jpg", "v3.jpg"])
    regs = lambda *rs: [{"shape_attributes": s, "region_attributes": a} for s, a in rs]
    d = {"_via_settings": {}, "_via_attributes": {"region": {"kind": {"type": "text"}, "label": {"type": "dropdown", "options": {"a": "", "b": ""}}}},
         "_via_img_metadata": {
             "v1.jpg1234": {"filename": "v1.jpg", "size": 1234, "file_attributes": {}, "regions": regs(
                 ({"name": "rect", "x": 10, "y": 10, "width": 20, "height": 30}, {"label": "a", "kind": "zzz"}),
                 ({"name": "polygon", "all_points_x": [0, 10, 10], "all_points_y": [0, 0, 10]}, {"label": "b"}),
                 ({"name": "circle", "cx": 50, "cy": 50, "r": 5}, {"label": "a"}),
                 ({"name": "ellipse", "cx": 80, "cy": 40, "rx": 10, "ry": 4}, {"label": "b"}),
                 ({"name": "point", "cx": 1, "cy": 1}, {"label": "a"}))},
             "v2.jpg99": {"filename": "v2.jpg", "size": 99, "file_attributes": {"scene": "indoor"}, "regions": {}},
             "v3.jpg5": {"filename": "v3.jpg", "size": 5, "file_attributes": {}, "regions": {"0": {"shape_attributes": {"name": "rect", "x": 1, "y": 2, "width": 3, "height": 4}, "region_attributes": {}}}}}}
    jdump(tmp / "via_project.json", d)
    ctx, det = top(tmp, "via")
    ps = parse(ctx, det, Options())
    r = {x.ref: x for x in ps.records}
    assert [o.kind for o in r["v1.jpg"].objs] == ["box", "poly", "box", "box"]
    assert [o.cls for o in r["v1.jpg"].objs] == ["a", "b", "a", "b"]
    assert r["v1.jpg"].objs[2].pts == [[45, 45], [55, 55]]
    assert r["v2.jpg"].tag == "indoor" and r["v3.jpg"].objs[0].cls == "object"
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 3 and rep.objects_added == 5, rep
    # старый плоский формат
    tmp2 = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp2, ["q.jpg"])
    jdump(tmp2 / "old.json", {"q.jpg100": {"filename": "q.jpg", "size": 100, "regions": [
        {"shape_attributes": {"name": "rect", "x": 1, "y": 1, "width": 10, "height": 10}, "region_attributes": {"type": "car"}}]}})
    ctx2, det2 = top(tmp2, "via")
    assert parse(ctx2, det2, Options()).records[0].objs[0].cls == "car"


def test_via_csv():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["c1.jpg", "c2.jpg"])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "file_size", "file_attributes", "region_count", "region_id", "region_shape_attributes", "region_attributes"])
    w.writerow(["c1.jpg", 10, "{}", 2, 0, json.dumps({"name": "rect", "x": 5, "y": 5, "width": 10, "height": 10}), json.dumps({"class": "cat", "x": "1"})])
    w.writerow(["c1.jpg", 10, "{}", 2, 1, json.dumps({"name": "polygon", "all_points_x": [1, 9, 9], "all_points_y": [1, 1, 9]}), json.dumps({"class": "dog"})])
    w.writerow(["c2.jpg", 20, "{}", 0, 0, "{}", "{}"])
    (tmp / "via.csv").write_text("﻿" + buf.getvalue(), encoding="utf-8")
    ctx, det = top(tmp, "via")
    ps = parse(ctx, det, Options())
    r = {x.ref: x for x in ps.records}
    assert [o.cls for o in r["c1.jpg"].objs] == ["cat", "dog"] and r["c2.jpg"].objs == []
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 2 and rep.objects_added == 2, rep


# ------------------------------------------------------------------ Label Studio
def test_labelstudio():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["l1.jpg", "l2.jpg", "l3.jpg"])
    res = lambda t, v, **kw: dict({"type": t, "value": v, "original_width": W, "original_height": H, "from_name": "x", "to_name": "image"}, **kw)
    tasks = [
        {"id": 1, "data": {"image": "/data/upload/3/ab12cd34-l1.jpg"}, "annotations": [
            {"was_cancelled": True, "result": [res("rectanglelabels", {"x": 0, "y": 0, "width": 5, "height": 5, "rectanglelabels": ["no"]})]},
            {"was_cancelled": False, "result": [
                res("rectanglelabels", {"x": 10, "y": 20, "width": 50, "height": 40, "rotation": 0, "rectanglelabels": ["cat"]}),
                res("polygonlabels", {"points": [[10, 10], [50, 10], [50, 50]], "polygonlabels": ["dog"]}),
                res("rectanglelabels", {"x": 50, "y": 50, "width": 20, "height": 10, "rotation": 90, "rectanglelabels": ["rot"]}),
                res("brushlabels", {"rle": [1, 2], "brushlabels": ["m"]}),
                res("choices", {"choices": ["sunny"]})]}]},
        {"id": 2, "data": {"image": "/data/local-files/?d=photos/l2.jpg"}, "annotations": [], "predictions": [
            {"result": [res("rectanglelabels", {"x": 0, "y": 0, "width": 50, "height": 50, "rectanglelabels": ["pred"]})]}]},
        {"id": 3, "data": {"img": "l3.jpg"}, "annotations": [{"result": [
            res("rectanglelabels", {"x": 0, "y": 0, "width": 10, "height": 10, "rectanglelabels": ["b"]}, id="r1"),
            res("keypointlabels", {"x": 1, "y": 2, "keypointlabels": ["nose"]}, id="k1", parent_id="r1"),
            res("keypointlabels", {"x": 3, "y": 4, "keypointlabels": ["eye"]}, id="k2", parent_id="r1")]}]},
        {"bad": "task"},
    ]
    jdump(tmp / "export.json", tasks)
    ctx, det = top(tmp, "labelstudio")
    ps = parse(ctx, det, Options())
    r = {x.ref: x for x in ps.records}
    assert set(r) == {"l1.jpg", "photos/l2.jpg", "l3.jpg"}, set(r)
    r1 = r["l1.jpg"]
    assert [o.kind for o in r1.objs] == ["box", "poly", "obb"] and r1.tag == "sunny" and r1.width == W
    assert r1.objs[0].norm and approx(r1.objs[0].pts[1][0], 0.6) and approx(r1.objs[0].pts[1][1], 0.6)
    # поворот на 90 град по часовой вокруг левого верхнего угла (в пикселях: 40,50 + повёрнутые 40x10)
    o = r1.objs[2].pts
    assert len(o) == 4
    assert approx(o[0][0] * W, 100) and approx(o[0][1] * H, 50) and approx(o[1][0] * W, 100) and approx(o[1][1] * H, 90), o
    assert r["photos/l2.jpg"].objs[0].cls == "pred"
    p3 = r["l3.jpg"].objs
    assert len(p3) == 1 and p3[0].kind == "pose" and p3[0].cls == "b" and ps.keypoints == ["nose", "eye"]
    assert any("маски" in n for n in ps.notes)
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 3 and rep.objects_added == 5 and rep.tags_set == 1, rep


def test_labelstudio_min_and_jsonl():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["m1.jpg", "m2.jpg"])
    mn = [{"image": "/data/upload/1/m1.jpg", "id": 1, "choice": "good", "label": [
        {"x": 10, "y": 10, "width": 20, "height": 20, "rotation": 0, "rectanglelabels": ["cat"], "original_width": W, "original_height": H}]},
          {"image": "m2.jpg", "id": 2, "label": [{"points": [[0, 0], [10, 0], [10, 10]], "polygonlabels": ["dog"], "original_width": W, "original_height": H}]}]
    jdump(tmp / "min.json", mn)
    ctx, det = top(tmp, "labelstudio")
    ps = parse(ctx, det, Options())
    assert [o.kind for o in ps.records[0].objs] == ["box"] and ps.records[0].tag == "good" and ps.records[1].objs[0].kind == "poly"
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 2 and rep.objects_added == 2, rep
    tmp2 = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp2, ["j.jpg"])
    line = {"id": 1, "data": {"image": "j.jpg"}, "annotations": [{"result": [
        {"type": "rectanglelabels", "original_width": W, "original_height": H, "value": {"x": 0, "y": 0, "width": 10, "height": 10, "rectanglelabels": ["a"]}}]}]}
    (tmp2 / "e.jsonl").write_text(json.dumps(line) + "\n" + json.dumps(line) + "\n", encoding="utf-8")
    ctx2, det2 = top(tmp2, "labelstudio")
    assert len(parse(ctx2, det2, Options()).records) == 2


# ------------------------------------------------------------------ CreateML
def test_createml():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    mk(tmp, ["k1.jpg", "k2.jpg"])
    d = [{"image": "k1.jpg", "annotations": [
        {"label": "cat", "coordinates": {"x": 50, "y": 40, "width": 20, "height": 10}},
        {"label": "dog", "coordinates": {"x": 100, "y": 50, "width": 30, "height": 30}}]},
         {"image": "k2.jpg", "annotations": []}, {"annotations": []}, 5]
    jdump(tmp / "annotations.json", d)
    ctx, det = top(tmp, "createml")
    ps = parse(ctx, det, Options())
    assert len(ps.records) == 2 and ps.notes
    assert ps.records[0].objs[0].pts == [[40, 35], [60, 45]] and ps.records[0].objs[1].cls == "dog"
    pr, rep = run(tmp, ctx, det)
    assert len(pr.images) == 2 and rep.objects_added == 2, rep


def test_no_cross_detection():
    # CreateML-файл не должен определяться как Label Studio / COCO и наоборот
    tmp = Path(tempfile.mkdtemp(prefix="mlide_aj_"))
    jdump(tmp / "a.json", [{"image": "x.jpg", "annotations": [{"label": "a", "coordinates": {"x": 1, "y": 1, "width": 2, "height": 2}}]}])
    names = [d.parser for d in detect(SourceContext([tmp]))]
    assert "labelstudio" not in names and "coco" not in names and "labelme" not in names, names


def main():
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL OK")


if __name__ == "__main__":
    main()
