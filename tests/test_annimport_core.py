"""Универсальный импорт разметки: ядро (поиск картинок, применение), YOLO, таблицы, файл-сопоставление, папки-классы."""
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MLIDE_SETTINGS_FILE"] = os.path.join(tempfile.gettempdir(), "mlide_test_annimport.ini")
import mlide.annimport as ai  # noqa: E402
from mlide.annimport import Options, SourceContext  # noqa: E402
from mlide.model import Project  # noqa: E402


def img(path, w=200, h=100, shade=120):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), shade, np.uint8))


def best(ctx):
    d = ai.detect(ctx)
    assert d, "формат не распознан"
    return d[0]


def run(paths, project=None, images_dir=None, opt=None):
    ctx = SourceContext(paths, images_dir)
    det = best(ctx)
    pr = project or Project("t")
    rep = ai.run_import(pr, ctx, det, opt or Options())
    return det, pr, rep


def test_yolo_layout_and_duplicates():
    d = Path(tempfile.mkdtemp())
    for sp in ("train", "val"):
        img(d / "images" / sp / "a.jpg", shade=50 if sp == "train" else 200)         # одно имя в двух частях
        (d / "labels" / sp).mkdir(parents=True)
    (d / "labels/train/a.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    (d / "labels/val/a.txt").write_text("1 0.5 0.5 0.5 0.5\n1 0.2 0.2 0.1 0.1\n")
    (d / "data.yaml").write_text("names:\n  0: cat\n  1: dog\n")
    det, pr, rep = run([d])
    assert det.parser == "yolo", det
    assert rep.images_added == 2 and rep.objects_added == 3, rep
    assert pr.class_names() == ["cat", "dog"]
    by = {Path(e.path).parent.name: e for e in pr.images}
    assert len(by["train"].anns) == 1 and len(by["val"].anns) == 2, "метки train/val не должны путаться"
    a = by["train"].anns[0]
    assert abs(a.pts[0][0] - 80) < 1 and abs(a.pts[1][0] - 120) < 1, a.pts
    print("ok yolo layout")


def test_labels_on_existing_images_and_separate_trees():
    d = Path(tempfile.mkdtemp())
    img(d / "pics" / "x.jpg")
    img(d / "pics" / "y.jpg")
    (d / "ann").mkdir()
    (d / "ann/x.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (d / "ann/y.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    (d / "ann/classes.txt").write_text("a\nb\n")
    pr = Project("t")
    pr.add_images([d / "pics" / "x.jpg", d / "pics" / "y.jpg"])                 # картинки уже в проекте
    det, pr, rep = run([d / "ann"], project=pr, images_dir=d / "pics")
    assert rep.images_added == 0 and rep.images_updated == 2 and rep.objects_added == 2, rep
    assert pr.class_names() == ["a", "b"]
    # режимы для уже размеченных
    ctx = SourceContext([d / "ann"], d / "pics")
    rep2 = ai.run_import(pr, ctx, best(ctx), Options(existing="skip"))
    assert rep2.images_skipped == 2 and rep2.objects_added == 0, rep2
    rep3 = ai.run_import(pr, ctx, best(ctx), Options(existing="replace"))
    assert rep3.objects_added == 2 and all(len(e.anns) == 1 for e in pr.images), rep3
    rep4 = ai.run_import(pr, ctx, best(ctx), Options(existing="append"))
    assert all(len(e.anns) == 2 for e in pr.images), rep4
    print("ok existing images / modes")


def test_table_variants():
    d = Path(tempfile.mkdtemp())
    img(d / "im" / "a.jpg", 200, 100)
    img(d / "im" / "b.png", 200, 100)
    # 1. TFOD CSV (пиксели)
    (d / "t1").mkdir()
    (d / "t1/labels.csv").write_text("filename,width,height,class,xmin,ymin,xmax,ymax\n"
                                     "a.jpg,200,100,cat,10,20,60,70\na.jpg,200,100,dog,100,10,150,90\nb.png,200,100,cat,0,0,50,50\n")
    det, pr, rep = run([d / "t1"], images_dir=d / "im")
    assert det.parser == "table" and rep.objects_added == 3 and pr.class_names() == ["cat", "dog"], (det, rep)
    e = next(x for x in pr.images if x.path.endswith("a.jpg"))
    assert e.anns[0].pts == [[10, 20], [60, 70]], e.anns[0].pts
    # 2. OpenImages: доли кадра, точка с запятой
    (d / "t2").mkdir()
    (d / "t2/boxes.csv").write_text("ImageID;LabelName;XMin;XMax;YMin;YMax\na;Car;0.1;0.5;0.2;0.8\n")
    det, pr, rep = run([d / "t2"], images_dir=d / "im")
    a = pr.images[0].anns[0]
    assert abs(a.pts[0][0] - 20) < 1e-6 and abs(a.pts[1][0] - 100) < 1e-6 and abs(a.pts[0][1] - 20) < 1e-6, a.pts
    # 3. без заголовка (Keras-RetinaNet): путь,x1,y1,x2,y2,класс
    (d / "t3").mkdir()
    (d / "t3/ann.csv").write_text("im/a.jpg,10,10,50,50,tree\nim/b.png,20,20,80,60,tree\n")
    det, pr, rep = run([d / "t3"], images_dir=d / "im")
    assert rep.objects_added == 2 and pr.class_names() == ["tree"], rep
    # 4. класс кадра: картинка + метка
    (d / "t4").mkdir()
    (d / "t4/cls.csv").write_text("image,label\na.jpg,sunny\nb.png,rainy\n")
    det, pr, rep = run([d / "t4"], images_dir=d / "im")
    assert rep.tags_set == 2 and sorted(pr.class_names()) == ["rainy", "sunny"], rep
    assert pr.class_names()[pr.images[0].tag] == "sunny"
    # 5. рамка в одной ячейке (xywh)
    (d / "t5").mkdir()
    (d / "t5/b.tsv").write_text("file\tbbox\tcategory\na.jpg\t[10, 20, 30, 40]\tcat\n")
    det, pr, rep = run([d / "t5"], images_dir=d / "im")
    assert pr.images[0].anns[0].pts == [[10, 20], [40, 60]], pr.images[0].anns[0].pts
    # 6. JSONL в стиле Hugging Face
    (d / "t6").mkdir()
    (d / "t6/meta.jsonl").write_text(json.dumps({"file_name": "a.jpg", "objects": {"bbox": [[10, 20, 30, 40], [0, 0, 20, 20]], "category": ["x", "y"]}}) + "\n")
    det, pr, rep = run([d / "t6"], images_dir=d / "im")
    assert det.parser == "table" and rep.objects_added == 2, (det, rep)
    # 7. ручное сопоставление столбцов через Options.params
    (d / "t7").mkdir()
    (d / "t7/weird.csv").write_text("pic,thing,aa,bb,cc,dd\na.jpg,box1,10,20,30,40\n")
    ctx = SourceContext([d / "t7"], d / "im")
    det = ai.detect(ctx)
    mapping = {"image": "pic", "class": "thing", "box": {"format": "xywh", "cols": ["aa", "bb", "cc", "dd"]}, "poly": None,
               "width": None, "height": None, "unit": "px"}
    pr = Project("t")
    rep = ai.run_import(pr, ctx, ai.Detection("table", 1.0, [d / "t7/weird.csv"]), Options(params={"mapping": mapping}))
    assert pr.images[0].anns[0].pts == [[10, 20], [40, 60]], rep
    print("ok table variants")


def test_pairs_file():
    d = Path(tempfile.mkdtemp())
    img(d / "photos" / "one.jpg")
    img(d / "photos" / "two.jpg")
    (d / "annots").mkdir()
    (d / "annots/L1.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    (d / "annots/L2.txt").write_text("0 0.3 0.3 0.2 0.2\n")
    (d / "classes.txt").write_text("apple\npear\n")
    (d / "map.csv").write_text("image,annotation\nphotos/one.jpg,annots/L1.txt\nphotos/two.jpg,annots/L2.txt\n")
    ctx = SourceContext([d])
    dets = ai.detect(ctx)
    assert dets[0].parser == "pairs", [(x.parser, x.score) for x in dets]
    pr = Project("t")
    rep = ai.run_import(pr, ctx, dets[0])
    assert rep.images_added == 2 and rep.objects_added == 2, rep
    one = next(e for e in pr.images if e.path.endswith("one.jpg"))
    assert pr.class_names()[one.anns[0].cls] == "pear", pr.class_names()
    print("ok pairs file")


def test_dotted_names():
    d = Path(tempfile.mkdtemp())
    img(d / "images/train/2026.rf.30a92a40e9e1.jpg")
    (d / "labels/train").mkdir(parents=True)
    (d / "labels/train/2026.rf.30a92a40e9e1.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    det, pr, rep = run([d])
    assert rep.images_added == 1 and rep.objects_added == 1 and not rep.unmatched, rep
    print("ok dotted names")


def test_short_windows_paths():
    """Каталоги с короткими именами 8.3 (Users/RUNNER~1, так на CI): путь картинки должен совпасть с тем, что хранит проект."""
    if sys.platform != "win32":
        return
    import ctypes
    d = Path(tempfile.mkdtemp()) / "a_rather_long_directory_name_for_testing"
    img(d / "images/train/some_long_image_name.jpg")
    (d / "labels/train").mkdir(parents=True)
    (d / "labels/train/some_long_image_name.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(str(d), buf, 1024)
    short = Path(buf.value)
    det, pr, rep = run([short])
    assert rep.images_added == 1 and rep.objects_added == 1, (short, rep)
    print("ok short paths")


def test_folders_and_nothing():
    d = Path(tempfile.mkdtemp())
    for c in ("cat", "dog"):
        for i in range(3):
            img(d / "train" / c / f"{i}.jpg")
    det, pr, rep = run([d])
    assert det.parser == "folders" and rep.tags_set == 6 and sorted(pr.class_names()) == ["cat", "dog"], (det, rep)
    e = tempfile.mkdtemp()
    assert ai.detect(SourceContext([e])) == []
    print("ok folders / empty")


if __name__ == "__main__":
    test_yolo_layout_and_duplicates()
    test_labels_on_existing_images_and_separate_trees()
    test_table_variants()
    test_pairs_file()
    test_dotted_names()
    test_short_windows_paths()
    test_folders_and_nothing()
    print("ALL OK")
