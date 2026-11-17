"""Computer-assisted labeling: YOLO ONNX (no torch needed), Ultralytics .pt, GrabCut smart polygon."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    cls: int
    conf: float
    box: tuple                      # x1,y1,x2,y2 in image pixels
    poly: list | None = None        # optional segmentation polygon
    name: str | None = None


def _letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class OnnxYolo:
    """Runs YOLOv5/v7/v8/v9/v11/v12 (anchor-free or v5 style) and end-to-end v10/v26 ONNX detectors."""

    def __init__(self, path, imgsz=640):
        # читаем байты сами: OpenCV не открывает файлы по путям с кириллицей на Windows
        with open(path, "rb") as f:
            self.net = cv2.dnn.readNetFromONNX(np.frombuffer(f.read(), np.uint8))
        self.imgsz = imgsz
        self.names = None

    def predict(self, img, conf=0.25, iou=0.45):
        lb, r, left, top = _letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()
        o = np.squeeze(out, 0) if out.ndim == 3 else out
        boxes, scores, cls_ids = [], [], []
        e2e = False
        if o.shape[0] < o.shape[1]:            # channels-first (4+nc, N): v8 / v11 / v12
            o = o.T
            mode = "v8"
        else:                                    # (N, C)
            mode = "e2e" if (o.shape[1] == 6 and o.shape[0] <= 1000) else "v5"
        if mode == "v8":
            sc = o[:, 4:]
            cid = sc.argmax(1)
            cf = sc.max(1)
            m = cf >= conf
            b = o[m, :4]
            xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
            cf, cid = cf[m], cid[m]
        elif mode == "v5":
            sc = o[:, 5:] * o[:, 4:5]
            cid = sc.argmax(1)
            cf = sc.max(1)
            m = cf >= conf
            b = o[m, :4]
            xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
            cf, cid = cf[m], cid[m]
        else:                                    # end-to-end: x1,y1,x2,y2,conf,cls
            m = o[:, 4] >= conf
            xyxy, cf, cid = o[m, :4], o[m, 4], o[m, 5].astype(int)
            e2e = True
        if len(cf) == 0:
            return []
        if not e2e:
            off = cid[:, None] * 10000.0          # per-class NMS
            rects = [[float(x1 + oo), float(y1), float(x2 - x1), float(y2 - y1)]
                     for (x1, y1, x2, y2), oo in zip(xyxy, off[:, 0])]
            keep = cv2.dnn.NMSBoxes(rects, cf.astype(float).tolist(), conf, iou)
            keep = np.array(keep).reshape(-1)
        else:
            keep = np.arange(len(cf))
        H, W = img.shape[:2]
        dets = []
        for k in keep:
            x1, y1, x2, y2 = xyxy[k]
            x1, x2 = (x1 - left) / r, (x2 - left) / r
            y1, y2 = (y1 - top) / r, (y2 - top) / r
            dets.append(Detection(int(cid[k]), float(cf[k]),
                                  (float(np.clip(x1, 0, W)), float(np.clip(y1, 0, H)),
                                   float(np.clip(x2, 0, W)), float(np.clip(y2, 0, H)))))
        return dets


class UltralyticsModel:
    def __init__(self, path, imgsz=640):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("The 'ultralytics' package is not installed.\n"
                               "Run:  pip install ultralytics\n"
                               "or export your model to ONNX and use the .onnx file.") from e
        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.names = self.model.names

    def predict(self, img, conf=0.25, iou=0.45):
        res = self.model.predict(img, conf=conf, iou=iou, imgsz=self.imgsz, verbose=False)[0]
        dets = []
        if res.boxes is None:
            return dets
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        polys = res.masks.xy if getattr(res, "masks", None) is not None else [None] * len(cls)
        for b, c, s, p in zip(xyxy, cls, cf, polys):
            dets.append(Detection(int(c), float(s), tuple(map(float, b)),
                                  p.tolist() if p is not None and len(p) >= 3 else None,
                                  self.names.get(int(c)) if isinstance(self.names, dict) else None))
        return dets


def resolve_weights(path) -> str:
    """.pt без установленного ultralytics (так в собранном приложении): берём готовый .onnx рядом или конвертируем
    системным Python, в котором есть ultralytics. Возвращает путь к модели, которую можно загрузить."""
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() != ".pt":
        return str(p)
    try:
        import ultralytics  # noqa: F401
        return str(p)
    except ImportError:
        pass
    for cand in (p.with_suffix(".onnx"), p.parent / "best.onnx", p.parent / "last.onnx"):
        if cand.exists() and cand.stat().st_mtime >= p.stat().st_mtime - 5:
            return str(cand)
    import shutil
    import subprocess
    import sys
    from . import paths
    py = shutil.which("python") or shutil.which("py")
    if not py:
        raise RuntimeError("Для файла .pt нужен Python с пакетом ultralytics, но Python не найден. Откройте «Результаты» → "
                           "«Экспорт модели…» и сделайте .onnx.")
    code = "from ultralytics import YOLO;import sys;print('OUT:'+str(YOLO(sys.argv[1]).export(format='onnx')))"
    with paths.child_launch():
        pr = subprocess.Popen([py, "-c", code, str(p)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", env=paths.child_env(),
                              creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0))
    out = pr.communicate(timeout=600)[0] or ""
    if pr.returncode != 0:
        tail = " ".join(out.strip().splitlines()[-2:])
        raise RuntimeError("Не удалось преобразовать .pt в .onnx (в Python нужен пакет ultralytics: Обучение → «Установить "
                           f"ultralytics»). {tail}")
    for ln in out.splitlines():
        if ln.startswith("OUT:") and Path(ln[4:].strip()).exists():
            return ln[4:].strip()
    if p.with_suffix(".onnx").exists():
        return str(p.with_suffix(".onnx"))
    raise RuntimeError("Преобразование в .onnx не создало файл. Сделайте экспорт вручную: «Результаты» → «Экспорт модели…».")


def load_model(path, imgsz=640):
    path = resolve_weights(path)
    if str(path).lower().endswith(".onnx"):
        return OnnxYolo(path, imgsz)
    return UltralyticsModel(path, imgsz)


# ------------------------------------------------------------------ smart polygon (GrabCut)
def smart_polygon(img, box, iterations=4, epsilon=0.004):
    """Rough box -> object outline. Returns polygon [[x,y],...] in image px, or None."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    pad = int(0.1 * max(x2 - x1, y2 - y1))
    cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    scale = 1.0
    if max(crop.shape[:2]) > 640:                    # keep GrabCut fast
        scale = 640 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rect = (int((x1 - cx1) * scale), int((y1 - cy1) * scale),
            int((x2 - x1) * scale), int((y2 - y1) * scale))
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * rect[2] * rect[3]:
        return None
    c = cv2.approxPolyDP(c, epsilon * cv2.arcLength(c, True), True)
    if len(c) < 3:
        return None
    return [[float(p[0][0] / scale + cx1), float(p[0][1] / scale + cy1)] for p in c]
