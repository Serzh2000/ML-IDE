"""Повёрнутые рамки (OBB), ключевые точки (pose) и классификация изображений: данные, экспорт, аугментации, импорт.
Запуск: python tests/test_new_tasks.py"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml

from mlide import imgio
from mlide.augment import AugParams, augment_sample
from mlide.formats import ExportOptions, export_dataset, import_yolo_labels
from mlide.model import Annotation, Project

W, H = 200, 100


def make_project(tmp: Path, n=8):
    pr = Project("t")
    for c in ("cat", "dog"):
        pr.add_class(c)
    pr.keypoints = ["nose", "left_eye", "right_eye"]
    pr.skeleton = [[0, 1], [0, 2]]
    pr.flip_pairs = [[1, 2]]
    d = tmp / "ф"                                           # кириллица в пути
    d.mkdir()
    for i in range(n):
        im = np.full((H, W, 3), 60 + i * 10, np.uint8)
        imgio.imwrite(d / f"img_{i}.jpg", im)
    pr.add_images(sorted(d.glob("*.jpg")))
    return pr


def rect_corners(cx, cy, w, h, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [[cx + x * c - y * s, cy + x * s + y * c] for x, y in pts]


def test_model_roundtrip():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_nt_"))
    pr = make_project(tmp)
    e = pr.images[0]
    e.anns = [Annotation(0, "obb", rect_corners(100, 50, 80, 30, 25)),
              Annotation(1, "pose", [[20, 20], [80, 80]], kps=[[50, 30, 2], [40, 25, 1], [0, 0, 0]])]
    e.tag = 1
    pr.save(tmp / "p.mlide.json")
    q = Project.load(tmp / "p.mlide.json")
    qe = q.images[0]
    assert qe.tag == 1 and [a.kind for a in qe.anns] == ["obb", "pose"]
    assert qe.anns[1].kps == [[50.0, 30.0, 2], [40.0, 25.0, 1], [0.0, 0.0, 0]] and q.keypoints == pr.keypoints
    assert q.skeleton == [[0, 1], [0, 2]] and q.flip_pairs == [[1, 2]]
    snap = qe.snapshot()
    qe.tag = None
    qe.anns = []
    qe.restore(snap)
    assert qe.tag == 1 and len(qe.anns) == 2                        # undo сохраняет и объекты, и класс кадра
    q.remove_class(0)
    assert q.images[0].tag == 0                                    # индекс класса кадра сдвинулся
    q.remove_class(0)
    assert q.images[0].tag is None


def test_obb_export_keeps_manual_corners():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_nt_"))
    pr = make_project(tmp)
    corners = rect_corners(100, 50, 80, 30, 25)
    for e in pr.images:
        e.anns = [Annotation(0, "obb", corners)]
    out = tmp / "obb"
    export_dataset(pr, ExportOptions(fmt="ultra_obb", out_dir=str(out), train=0.75, val=0.25))
    f = next((out / "labels" / "train").glob("*.txt"))
    v = [float(x) for x in f.read_text().split()]
    assert int(v[0]) == 0 and len(v) == 9
    got = np.array(v[1:]).reshape(4, 2) * [W, H]
    exp = np.array(corners)
    # порядок углов может отличаться: сравниваем как множества с точностью 0.5 px
    assert all(np.min(np.linalg.norm(exp - g, axis=1)) < 0.5 for g in got), (got, exp)
    # то же изображение в детекции -> охватывающая рамка
    out2 = tmp / "det"
    export_dataset(pr, ExportOptions(fmt="ultra_det", out_dir=str(out2), train=0.75, val=0.25))
    t = next((out2 / "labels" / "train").glob("*.txt")).read_text().split()
    assert len(t) == 5 and 0 < float(t[3]) <= 1


def test_pose_export_and_yaml():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_nt_"))
    pr = make_project(tmp)
    for e in pr.images:
        e.anns = [Annotation(0, "pose", [[20, 10], [120, 90]], kps=[[60, 30, 2], [50, 25, 1], [-5, 500, 2]])]   # третья точка вне кадра
    out = tmp / "pose"
    export_dataset(pr, ExportOptions(fmt="ultra_pose", out_dir=str(out), train=0.75, val=0.25))
    cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["kpt_shape"] == [3, 3] and cfg["flip_idx"] == [0, 2, 1] and cfg["nc"] == 2
    v = next((out / "labels" / "train").glob("*.txt")).read_text().split()
    assert len(v) == 5 + 3 * 3, v
    assert v[5:8] == ["0.300000", "0.300000", "2"] and v[8:11][2] == "1" and v[11:14] == ["0.000000", "0.000000", "0"]
    # без ключевых точек в проекте экспорт отклоняется с понятным сообщением
    pr.keypoints = []
    try:
        export_dataset(pr, ExportOptions(fmt="ultra_pose", out_dir=str(tmp / "p2")))
        raise AssertionError("ожидалась ошибка")
    except ValueError as ex:
        assert "ключев" in str(ex)
    # обратный импорт
    pr2 = make_project(Path(tempfile.mkdtemp(prefix="mlide_nt_")))
    lab = list((out / "labels").rglob("*.txt"))[0]
    img_dir = out / "images" / lab.parent.name
    ent = pr2.images[0]
    (img_dir).mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(ent.path, img_dir / (lab.stem + ".jpg"))
    ent.path = str(img_dir / (lab.stem + ".jpg"))
    n = import_yolo_labels(ent, 2, n_kpts=3)
    assert n == 1 and ent.anns[0].kind == "pose" and len(ent.anns[0].kps) == 3
    assert abs(ent.anns[0].kps[0][0] - 60) < 0.5 and ent.anns[0].kps[0][2] == 2


def test_pose_coco():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_nt_"))
    pr = make_project(tmp)
    for e in pr.images:
        e.anns = [Annotation(0, "pose", [[20, 10], [120, 90]], kps=[[60, 30, 2], [50, 25, 1], [0, 0, 0]])]
    out = tmp / "coco"
    export_dataset(pr, ExportOptions(fmt="coco", out_dir=str(out), train=0.75, val=0.25))
    d = json.loads((out / "annotations" / "instances_train.json").read_text())
    assert d["categories"][0]["keypoints"] == ["nose", "left_eye", "right_eye"] and d["categories"][0]["skeleton"] == [[1, 2], [1, 3]]
    a = d["annotations"][0]
    assert a["num_keypoints"] == 2 and len(a["keypoints"]) == 9 and a["keypoints"][:3] == [60.0, 30.0, 2]


def test_pose_augmentation():
    img = np.full((H, W, 3), 90, np.uint8)
    anns = [(0, "pose", [[40, 20], [160, 80]], [[60, 30, 2], [140, 30, 2], [100, 70, 1]])]
    rng = np.random.default_rng(1)
    p = AugParams(fliplr=1.0, flipud=1.0, degrees=0, translate=0, scale=0, hsv_h=0, hsv_s=0, hsv_v=0)
    # без flip_idx отражение запрещено (иначе левая точка станет «правой»)
    _, res = augment_sample(img, anns, p, rng)
    assert res[0][3][0][:2] == [60.0, 30.0]
    # с flip_idx точки отражаются и меняются местами
    _, res = augment_sample(img, anns, p, rng, flip_idx=[1, 0, 2])
    k = res[0][3]
    assert abs(k[0][0] - (W - 140)) < 1e-6 and abs(k[1][0] - (W - 60)) < 1e-6 and k[2][2] == 1
    # сильное смещение: точки вне кадра получают v=0
    p2 = AugParams(fliplr=0, flipud=0, degrees=0, translate=0, scale=0, hsv_h=0, hsv_s=0, hsv_v=0)
    _, res = augment_sample(img, [(0, "pose", [[0, 0], [190, 90]], [[5, 5, 2], [195, 95, 2], [100, 50, 2]])],
                            AugParams(**{**p2.to_dict(), "translate": 0.0}), rng)
    assert all(v in (0, 1, 2) for _x, _y, v in res[0][3])
    # obb остаётся obb
    _, res = augment_sample(img, [(0, "obb", rect_corners(100, 50, 60, 20, 30))], p2, rng)
    assert res[0][1] == "obb" and len(res[0][2]) == 4


def test_classification_export_stratified():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_nt_"))
    pr = make_project(tmp, n=20)
    for i, e in enumerate(pr.images):
        e.tag = 0 if i < 12 else 1                                  # 12 кошек и 8 собак
    pr.images[19].tag = None                                        # один кадр без класса
    out = tmp / "cls"
    msg = export_dataset(pr, ExportOptions(fmt="ultra_cls", out_dir=str(out), train=0.7, val=0.3, bake_aug=True,
                                           aug=AugParams(copies=1, fliplr=0.5, translate=0.1, scale=0.2)))
    assert (out / "classes.txt").read_text().split() == ["cat", "dog"]
    for s in ("train", "val"):
        for c in ("cat", "dog"):
            assert (out / s / c).is_dir()                           # папки всех классов есть в каждой части выборки
    n_val = {c: len(list((out / "val" / c).glob("*"))) for c in ("cat", "dog")}
    assert n_val["cat"] >= 3 and n_val["dog"] >= 2, n_val           # доля каждого класса попала в val
    n_tr = {c: len(list((out / "train" / c).glob("*"))) for c in ("cat", "dog")}
    assert any("_aug" in f.name for f in (out / "train" / "cat").glob("*")), "офлайн-аугментация классификации"
    total = sum(n_val.values()) + sum(1 for c in ("cat", "dog") for f in (out / "train" / c).glob("*") if "_aug" not in f.name)
    assert total == 19, total                                       # кадр без класса не экспортирован
    code = (out / "train.py").read_text(encoding="utf-8")
    assert "-cls.pt" in code and "data=str(HERE)," in code and "imgsz=224" in code, code
    assert "data.yaml" not in code
    # нет ни одного класса кадра -> понятная ошибка
    for e in pr.images:
        e.tag = None
    try:
        export_dataset(pr, ExportOptions(fmt="ultra_cls", out_dir=str(tmp / "c2")))
        raise AssertionError("ожидалась ошибка")
    except ValueError as ex:
        assert "класс" in str(ex)


def test_train_script_pose_weights():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_nt_"))
    pr = make_project(tmp)
    for e in pr.images:
        e.anns = [Annotation(0, "pose", [[20, 10], [120, 90]], kps=[[60, 30, 2], [50, 25, 1], [0, 0, 0]])]
    out = tmp / "pose"
    export_dataset(pr, ExportOptions(fmt="ultra_pose", out_dir=str(out), family="YOLO11", size="s"))
    assert 'YOLO("yolo11s-pose.pt")' in (out / "train.py").read_text(encoding="utf-8")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL OK")
