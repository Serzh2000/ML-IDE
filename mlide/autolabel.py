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
    poly: list | None = None        # optional segmentation polygon (или 4 угла повёрнутой рамки при obb=True)
    name: str | None = None
    kps: list | None = None         # ключевые точки [[x, y, v], ...] (pose)
    obb: bool = False               # poly - повёрнутая рамка


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
    """YOLO ONNX (Ultralytics v5/v8/v9/v11/v12, end-to-end v10/v26): детекция, сегментация, повёрнутые рамки, ключевые точки, классификация.

    Задача и классы берутся из метаданных ONNX (их пишет экспорт Ultralytics: task, names, kpt_shape, imgsz); если метаданных нет,
    задача определяется по числу и форме выходов. Размер входа - тот, что зашит в модель при экспорте (а не из поля в окне):
    несовпадение размера раньше ломало разбор выходов.
    """

    def __init__(self, path, imgsz=640):
        import ast
        import onnxruntime as ort
        with open(path, "rb") as f:                              # байты, а не путь: кириллица в пути
            raw = f.read()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        self.size = (int(shp[2]), int(shp[3])) if len(shp) == 4 and isinstance(shp[2], int) and isinstance(shp[3], int) else (imgsz, imgsz)
        self.imgsz = self.size[0]
        meta = {}
        try:
            meta = dict(self.sess.get_modelmeta().custom_metadata_map or {})
        except Exception:  # noqa
            pass
        self.task = (meta.get("task") or "").lower() or None
        self.names = None
        try:
            nm = ast.literal_eval(meta["names"]) if "names" in meta else None
            if isinstance(nm, dict):
                self.names = [str(nm[k]) for k in sorted(nm)]
            elif isinstance(nm, (list, tuple)):
                self.names = [str(x) for x in nm]
        except Exception:  # noqa
            pass
        self.kpt_shape = None
        try:
            ks = ast.literal_eval(meta["kpt_shape"]) if "kpt_shape" in meta else None
            if ks and len(ks) == 2:
                self.kpt_shape = (int(ks[0]), int(ks[1]))
        except Exception:  # noqa
            pass

    # ------------------------------------------------------------------ разбор
    def _kind(self, outs):
        """det | seg | pose | obb | cls | e2e - по метаданным, а при их отсутствии по выходам."""
        o0 = outs[0]
        if self.task in ("segment", "pose", "obb", "classify", "detect"):
            kind = {"segment": "seg", "classify": "cls", "detect": "det"}.get(self.task, self.task)
        else:
            kind = None
        if kind is None:
            if len(outs) >= 2 and outs[1].ndim == 4:
                kind = "seg"
            elif o0.ndim == 2 and o0.shape[0] == 1 and o0.shape[1] <= 5000:
                kind = "cls"
            else:
                kind = "det"
        if kind == "det":                                         # end-to-end (v10/v26): (1, 300, 6), числа по строкам
            a = o0[0] if o0.ndim == 3 else o0
            if a.ndim == 2 and a.shape[1] in (6, 7) and a.shape[0] <= 1000 and a.shape[0] != a.shape[1]:
                kind = "e2e"
        return kind

    def predict(self, img, conf=0.25, iou=0.45):
        H, W = img.shape[:2]
        sh, sw = self.size
        lb, r, left, top = _letterbox(img, sw if sw == sh else max(sw, sh))
        if lb.shape[0] != sh or lb.shape[1] != sw:
            lb = cv2.resize(lb, (sw, sh))
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (sw, sh), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob.astype(np.float32)})
        kind = self._kind(outs)
        if kind == "cls":
            p = np.asarray(outs[0]).reshape(-1)
            k = int(p.argmax())
            c = float(p[k])
            if c < conf:
                return []
            return [Detection(k, c, (0.0, 0.0, float(W), float(H)), None, self.names[k] if self.names and k < len(self.names) else None)]
        o = np.asarray(outs[0])
        o = o[0] if o.ndim == 3 else o
        if kind == "e2e":
            m = o[:, 4] >= conf
            res = []
            for row in o[m]:
                x1, y1, x2, y2 = row[:4]
                res.append(self._mk(int(row[5]), float(row[4]), ((x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r), W, H))
            return res
        extra = 0
        protos = None
        if kind == "seg":
            protos = np.asarray(outs[1])[0]                       # (nm, mh, mw)
            extra = protos.shape[0]
        elif kind == "pose":
            ks = self.kpt_shape or (17, 3)
            extra = ks[0] * ks[1]
        elif kind == "obb":
            extra = 1
        # какая ось - «каналы» (4 + классы + дополнительные), а какая - «якоря»: по известному числу каналов, иначе по меньшей оси
        # (правило «меньшая ось» одно подводит при очень малом входе и многих классах: якорей меньше, чем каналов)
        chan = (4 + len(self.names) + extra) if self.names else None
        transposed = True
        if chan is not None and o.shape[0] == chan and o.shape[1] != chan:
            o = o.T
        elif chan is not None and o.shape[1] == chan:
            transposed = False
        elif o.shape[0] < o.shape[1]:
            o = o.T
        else:
            transposed = False
        C = o.shape[1]
        # старый формат YOLOv5/v7 (без метаданных, выход уже (якоря, 5 + классы)): есть столбец «объектность», оценка = объектность * класс
        v5 = (not transposed) and self.task is None and kind == "det" and not (chan is not None and C == chan)
        nc = C - 5 if v5 else C - 4 - extra
        if self.names and len(self.names) <= nc and kind in ("det", "obb"):
            nc = len(self.names)
        if nc <= 0:
            return []
        sc = o[:, 5:5 + nc] * o[:, 4:5] if v5 else o[:, 4:4 + nc]
        cid = sc.argmax(1)
        cf = sc.max(1)
        m = cf >= conf
        if not m.any():
            return []
        o, cid, cf = o[m], cid[m], cf[m]
        cx, cy, bw, bh = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        off = cid.astype(np.float64) * 10000.0                    # NMS отдельно по классам
        if kind == "obb":
            ang = o[:, 4 + nc]
            rects = [((float(x), float(y)), (float(w), float(h)), float(np.degrees(a))) for x, y, w, h, a in zip(cx + off, cy, bw, bh, ang)]
            keep = cv2.dnn.NMSBoxesRotated(rects, cf.astype(float).tolist(), conf, iou)
        else:
            rects = [[float(x - w / 2 + oo), float(y - h / 2), float(w), float(h)] for x, y, w, h, oo in zip(cx, cy, bw, bh, off)]
            keep = cv2.dnn.NMSBoxes(rects, cf.astype(float).tolist(), conf, iou)
        keep = np.array(keep).reshape(-1).astype(int)
        res = []
        for k in keep[:300]:
            x1, y1, x2, y2 = cx[k] - bw[k] / 2, cy[k] - bh[k] / 2, cx[k] + bw[k] / 2, cy[k] + bh[k] / 2
            box = ((x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r)
            d = self._mk(int(cid[k]), float(cf[k]), box, W, H)
            if kind == "seg":
                d.poly = self._mask_poly(o[k, 4 + nc:4 + nc + extra], protos, (x1, y1, x2, y2), (left, top, r), (W, H), sh, sw)
            elif kind == "pose":
                n_k, dim = self.kpt_shape or (17, 3)
                kp = o[k, 4 + nc:4 + nc + extra].reshape(n_k, dim)
                d.kps = [[float(np.clip((x - left) / r, 0, W)), float(np.clip((y - top) / r, 0, H)),
                          (2 if (dim < 3 or v >= 0.5) else 0)] for x, y, *rest in kp for v in [rest[0] if rest else 1.0]]
            elif kind == "obb":
                pts = cv2.boxPoints(((float(cx[k]), float(cy[k])), (float(bw[k]), float(bh[k])), float(np.degrees(o[k, 4 + nc]))))
                d.poly = [[float(np.clip((x - left) / r, 0, W)), float(np.clip((y - top) / r, 0, H))] for x, y in pts]
                d.obb = True
                xs, ys = [q[0] for q in d.poly], [q[1] for q in d.poly]
                d.box = (min(xs), min(ys), max(xs), max(ys))                # охватывающая рамка повёрнутой (для сравнения и показа)
            res.append(d)
        return res

    def _mk(self, c, s, box, W, H):
        x1, y1, x2, y2 = box
        nm = self.names[c] if self.names and 0 <= c < len(self.names) else None
        return Detection(c, s, (float(np.clip(x1, 0, W)), float(np.clip(y1, 0, H)), float(np.clip(x2, 0, W)), float(np.clip(y2, 0, H))), None, nm)

    @staticmethod
    def _mask_poly(coef, protos, box_lb, lbinfo, wh, sh, sw):
        """Маска объекта: коэффициенты x прототипы -> сигмоида -> обрезка по рамке -> контур в координатах исходной картинки."""
        left, top, r = lbinfo
        W, H = wh
        nm, mh, mw = protos.shape
        m = 1.0 / (1.0 + np.exp(-(coef @ protos.reshape(nm, -1)).reshape(mh, mw)))
        m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = [int(round(v)) for v in box_lb]
        crop = np.zeros_like(m)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(sw, x2), min(sh, y2)
        crop[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        # убираем поля letterbox и возвращаем размер исходной картинки
        nw, nh = int(round(W * r)), int(round(H * r))
        t, l = int(round(top)), int(round(left))
        crop = crop[t:t + nh, l:l + nw]
        if crop.size == 0:
            return None
        mask = (cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 4:
            return None
        c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)
        return [[float(x), float(y)] for x, y in c.reshape(-1, 2)] if len(c) >= 3 else None


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
