"""Разбор выводов ONNX (детекция, сегментация, повёрнутые рамки, ключевые точки, классификация) совпадает с Ultralytics.

Регрессия: сегментационную модель раньше разбирали как детектор - на кадре получались сотни ложных «находок», картинка заливалась цветом.
Нужен установленный ultralytics (в CI его нет - тест пропускается). Модели создаются из yaml без предобучения (веса случайные),
для сравнения важна сама раскладка выводов, а не качество.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlide import autolabel  # noqa: E402

try:
    from ultralytics import YOLO
except Exception:  # noqa
    print("ultralytics не установлен: тест пропущен")
    print("ALL OK")
    sys.exit(0)

CASES = [("yolo11n.yaml", "detect"), ("yolo11n-seg.yaml", "segment"), ("yolo11n-pose.yaml", "pose"), ("yolo11n-obb.yaml", "obb"),
         ("yolo11n-cls.yaml", "classify")]


def check(cfg, task, work):
    os.chdir(work)
    m = YOLO(cfg)
    onnx = Path(m.export(format="onnx", imgsz=64, simplify=False, verbose=False))
    ref = YOLO(str(onnx), task=task)
    ours = autolabel.OnnxYolo(str(onnx), 64)
    assert ours.task == task, (ours.task, task)
    rng = np.random.default_rng(1)
    img = (rng.random((90, 120, 3)) * 255).astype(np.uint8)              # шум: случайная модель даёт много слабых «находок»
    r = ref.predict(img, conf=0.01, iou=0.45, imgsz=64, verbose=False)[0]
    d = ours.predict(img, 0.0 if task == "classify" else 0.01, 0.45)
    if task == "classify":
        assert d and d[0].cls == int(r.probs.top1), (d[:1], r.probs.top1)
        return
    if task == "obb":
        rb = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        rc = r.obb.conf.cpu().numpy()
    else:
        rb = r.boxes.xyxy.cpu().numpy()
        rc = r.boxes.conf.cpu().numpy()
    top = np.argsort(-rc)[:3]
    assert len(d) > 0 or len(rc) == 0, "ours found nothing where Ultralytics did"
    ours_sorted = sorted(d, key=lambda x: -x.conf)[:3]
    for k, x in zip(top, ours_sorted):
        assert abs(x.conf - float(rc[k])) < 5e-3, (task, x.conf, rc[k])
        if task == "obb":
            c = np.array(x.poly)
            assert np.abs(np.sort(c, 0) - np.sort(rb[k], 0)).max() < 1.5, (task, c, rb[k])
        else:
            assert np.abs(np.array(x.box) - rb[k]).max() < 1.0, (task, x.box, rb[k])
    if task == "segment":
        assert all(x.poly is None or len(x.poly) >= 3 for x in d)            # у случайной модели маски пустые - важна форма
    if task == "pose":
        assert all(x.kps and len(x.kps) == 17 for x in d)
    # главное для регрессии: число находок разумное (не сотни, как у детектора, читающего чужие выводы)
    assert len(d) <= max(3 * len(rc), 20), (task, len(d), len(rc))


def main():
    work = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        for cfg, task in CASES:
            check(cfg, task, work)
            print("ok", task)
    finally:
        os.chdir(cwd)
    print("ALL OK")


if __name__ == "__main__":
    main()
