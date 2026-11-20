"""Парсеры voc / cvat / txtper универсального импорта разметки. Запуск: python tests/test_annimport_xmltxt.py"""
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from mlide.annimport import Options, SourceContext, detect, parse, run_import
from mlide.model import Project

W, H = 200, 100


def tmpdir():
    return Path(tempfile.mkdtemp(prefix="mlide_xt_"))


def img(path: Path, w=W, h=H):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", np.full((h, w, 3), 90, np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def top(ctx):
    d = detect(ctx)
    return d[0] if d else None


def find(ctx, name):
    return next((d for d in detect(ctx) if d.parser == name), None)


# ---------------------------------------------------------------- VOC
VOC = """<?xml version="1.0" encoding="utf-8"?>
<annotation><folder>x</folder><filename>{fn}</filename><path>C:\\old\\dir\\{fn}</path>
<size><width>200</width><height>100</height><depth>3</depth></size>
{objs}</annotation>"""
VOC_OBJ = "<object><name>{n}</name><difficult>0</difficult><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"


def make_voc(root: Path):
    img(root / "JPEGImages" / "a.jpg")
    img(root / "JPEGImages" / "b.jpg")
    img(root / "JPEGImages" / "c.jpg")
    ann = root / "Annotations"
    ann.mkdir(parents=True)
    (ann / "a.xml").write_text(VOC.format(fn="a.jpg", objs=VOC_OBJ.format(n="cat", a=11, b=21, c=60, d=80)
                                           + VOC_OBJ.format(n="dog", a=100, b=10, c=150, d=50.5)), encoding="utf-8")
    poly = ("<object><name>blob</name><polygon><x1>10</x1><y1>10</y1><x2>50</x2><y2>10</y2><x3>30</x3><y3>40</y3></polygon></object>"
            "<object><name>rot</name><robndbox><cx>100</cx><cy>50</cy><w>40</w><h>20</h><angle>0</angle></robndbox></object>")
    (ann / "b.xml").write_text(VOC.format(fn="b.jpg", objs=poly), encoding="utf-8-sig")
    (ann / "c.xml").write_text(VOC.format(fn="c.jpg", objs=""), encoding="utf-8")           # фон
    (ann / "broken.xml").write_text("<annotation><filename>zzz", encoding="utf-8")


def test_voc():
    root = tmpdir()
    make_voc(root)
    ctx = SourceContext([root])
    d = top(ctx)
    assert d and d.parser == "voc" and d.score >= 0.8, d
    pr = parse(ctx, d, Options())
    assert len(pr.records) == 3 and any("broken.xml" in n for n in pr.notes), (len(pr.records), pr.notes)
    a = next(r for r in pr.records if r.ref == "a.jpg")
    assert (a.width, a.height) == (200, 100) and "a" in a.extra_refs
    assert a.objs[0].kind == "box" and a.objs[0].cls == "cat" and a.objs[0].pts == [[10, 20], [60, 80]], a.objs[0]
    assert a.objs[1].pts == [[99, 9], [150, 50.5]]
    b = next(r for r in pr.records if r.ref == "b.jpg")
    assert [o.kind for o in b.objs] == ["poly", "obb"] and len(b.objs[0].pts) == 3
    assert b.objs[1].pts[0] == [80.0, 40.0] and b.objs[1].pts[2] == [120.0, 60.0]
    assert next(r for r in pr.records if r.ref == "c.jpg").objs == []
    p = Project()
    rep = run_import(p, ctx, d, Options())
    assert rep.images_added == 3 and rep.objects_added == 4, rep.summary()
    assert sorted(c["name"] for c in p.classes) == ["blob", "cat", "dog", "rot"]


# ---------------------------------------------------------------- CVAT
CVAT = """<?xml version="1.0" encoding="utf-8"?>
<annotations><version>1.1</version><meta><task><labels><label><name>car</name></label></labels></task></meta>
<image id="0" name="images/one.jpg" width="200" height="100">
  <box label="car" occluded="0" xtl="10" ytl="20" xbr="60" ybr="40" z_order="0"></box>
  <box label="car" occluded="0" xtl="80" ytl="40" xbr="120" ybr="60" rotation="90"></box>
  <polygon label="road" points="5,5;50,5;50,50;5,50"></polygon>
  <points label="pt" points="10,10"></points>
  <polyline label="ln" points="1,1;5,5"></polyline>
  <tag label="day"></tag><tag label="night"></tag>
</image>
<image id="1" name="two.jpg" width="200" height="100"><box label="car" xtl="1" ytl="1" xbr="30" ybr="30"/></image>
<image id="2" name="three.jpg" width="200" height="100"><polygon label="road" points="zz"/></image>
</annotations>"""


def test_cvat():
    root = tmpdir()
    for n in ("images/one.jpg", "two.jpg", "three.jpg"):
        img(root / n)
    (root / "annotations.xml").write_text(CVAT, encoding="utf-8")
    ctx = SourceContext([root])
    d = top(ctx)
    assert d and d.parser == "cvat", detect(ctx)
    assert find(ctx, "voc") is None                    # voc не должен узнавать annotations.xml
    pr = parse(ctx, d, Options())
    assert len(pr.records) == 3
    one = pr.records[0]
    assert one.ref == "images/one.jpg" and one.tag == "day" and (one.width, one.height) == (200, 100)
    assert [o.kind for o in one.objs] == ["box", "obb", "poly"]
    got = sorted(map(tuple, (np.round(one.objs[1].pts, 3)).tolist()))
    # рамка 40x20 с центром (100,50), поворот на 90 градусов -> 20x40
    assert got == sorted([(110.0, 30.0), (110.0, 70.0), (90.0, 30.0), (90.0, 70.0)]), got
    assert one.objs[1].pts[0] == [110.0, 30.0], one.objs[1].pts                   # левый верхний угол -> правый верхний
    assert any("points" in n for n in pr.notes) and any("polyline" in n for n in pr.notes)
    p = Project()
    rep = run_import(p, ctx, d, Options())
    assert rep.images_added == 3 and rep.objects_added == 4 and rep.tags_set == 1, rep.summary()


# ---------------------------------------------------------------- txtper
def _fill(root: Path, labels: dict, ext=".jpg", sub="labels"):
    for stem, text in labels.items():
        img(root / "images" / (stem + ext))
        f = root / sub / (stem + ".txt")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")


def _run(root, expect_kind, n_obj, **kw):
    ctx = SourceContext([root])
    d = top(ctx)
    assert d and d.parser == "txtper", detect(ctx)
    pr = parse(ctx, d, Options())
    assert all(o.kind == expect_kind for r in pr.records for o in r.objs)
    assert sum(len(r.objs) for r in pr.records) == n_obj, pr
    p = Project()
    rep = run_import(p, ctx, d, Options())
    assert rep.images_added == len(pr.records) and rep.objects_added == n_obj, rep.summary()
    return ctx, d, pr, p


def test_dota():
    root = tmpdir()
    _fill(root, {f"p{i}": "imagesource:GoogleEarth\ngsd:0.1\n10 10 60 10 60 40 10 40 plane 0\n100 50 150 50 150 90 100 90 small-vehicle 1\n"
                 for i in range(4)})
    ctx, d, pr, p = _run(root, "obb", 8)
    assert pr.records[0].objs[0].cls == "plane" and pr.records[0].objs[0].pts[2] == [60.0, 40.0]
    assert sorted(c["name"] for c in p.classes) == ["plane", "small-vehicle"]


def test_kitti():
    root = tmpdir()
    line = "Car 0.00 0 -1.58 {a} {b} {c} {d} 1.5 1.6 3.9 1 2 3 0.1\n"
    _fill(root, {f"k{i}": line.format(a=10, b=20, c=100, d=80) + "DontCare -1 -1 -10 5 5 20 20 -1 -1 -1 -1000 -1000 -1000 -10\n"
                 + "Pedestrian 0 0 0 30 30 60 90 1 1 1 0 0 0 0\n" for i in range(3)})
    ctx, d, pr, p = _run(root, "box", 6)
    assert d.score >= 0.9
    assert pr.records[0].objs[0].pts == [[10, 20], [100, 80]] and pr.records[0].objs[0].cls == "Car"


def test_xyxy_class_first_with_names():
    root = tmpdir()
    _fill(root, {"a": "0 10 20 60 80\n1 100 10 150 50\n", "b": "1 5 5 40 40\n", "c": "0 20 20 90 90\n"})
    (root / "labels" / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    ctx, d, pr, p = _run(root, "box", 4)
    assert pr.class_names == ["cat", "dog"] and isinstance(pr.records[0].objs[0].cls, int)
    assert sorted(c["name"] for c in p.classes) == ["cat", "dog"]
    assert len(pr.records) == 3                          # classes.txt не стал записью


def test_xyxy_class_last_words_and_bad_file():
    root = tmpdir()
    _fill(root, {f"a{i}": "10 20 60 80 cat\n100 10 150 50.5 dog\n" for i in range(5)})
    (root / "labels" / "a2.txt").write_text("10 20 60 80 cat\nthis is garbage\n", encoding="utf-8")
    (root / "labels" / "notes.txt").write_text("hello world\n", encoding="utf-8")
    ctx = SourceContext([root])
    d = top(ctx)
    assert d and d.parser == "txtper"
    pr = parse(ctx, d, Options())
    assert len(pr.records) == 5 and any("a2.txt" in n for n in pr.notes), pr.notes
    assert sum(len(r.objs) for r in pr.records) == 9
    assert pr.records[0].objs[0].cls == "cat" and pr.records[0].objs[1].pts == [[100, 10], [150, 50.5]]


def test_xywh_and_norm_word():
    root = tmpdir()
    _fill(root, {"a": "cat 10 20 50 60\ndog 100 10 40 30\n"})        # x2 <= x1 в строке 2 -> xywh
    ctx, d, pr, p = _run(root, "box", 2)
    assert d.params["mode"] == "xywh" and pr.records[0].objs[1].pts == [[100, 10], [140, 40]]
    root = tmpdir()
    _fill(root, {"a": "cat 0.1 0.2 0.5 0.8\n", "b": "dog 0.2 0.2 0.6 0.9\n"})
    ctx, d, pr, p = _run(root, "box", 2)
    assert pr.records[0].objs[0].norm is True
    a = p.images[0].anns[0]
    assert [round(v) for p_ in a.pts for v in p_] == [20, 20, 100, 80], a.pts


def test_negative():
    root = tmpdir()
    _fill(root, {f"y{i}": "0 0.5 0.5 0.2 0.3\n1 0.3 0.3 0.1 0.1\n" for i in range(5)})
    ctx = SourceContext([root])
    assert find(ctx, "txtper") is None and find(ctx, "voc") is None
    # YOLO-seg / лишние колонки тоже не наши
    root = tmpdir()
    _fill(root, {"s": "0 0.1 0.1 0.5 0.1 0.5 0.5\n"})
    assert find(SourceContext([root]), "txtper") is None


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL OK")


if __name__ == "__main__":
    main()
