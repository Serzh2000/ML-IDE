"""Разметка по клику: MobileSAM (ONNX, onnxruntime). Клик по объекту -> маска -> полигон/рамка.

Модель (≈43 МБ) скачивается один раз с Hugging Face (Acly/MobileSAM) в папку пользователя.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import paths

# закреплённая версия репозитория (не «main»): содержимое файлов нельзя подменить после выхода релиза, плюс проверка SHA-256
REPO = "https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8/"
FILES = {"mobile_sam_image_encoder.onnx": 26_000_000, "sam_mask_decoder_single.onnx": 15_000_000}   # минимальные размеры
SHA256 = {"mobile_sam_image_encoder.onnx": "580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749",
          "sam_mask_decoder_single.onnx": "93915fc7c993ab9d59ab8c9ccd3bce37f7509c81ab4150a74abd4d2abbd8570d"}
LICENSE_NOTE = "MobileSAM (Apache-2.0), экспорт в ONNX: Acly/MobileSAM на Hugging Face."


def models_dir() -> Path:
    import os
    env = os.environ.get("MLIDE_MODELS")        # для тестов: не трогать домашнюю папку пользователя
    return Path(env) if env else paths.examples_dir().parent / "models"


def models_ready() -> bool:
    d = models_dir()
    return all((d / n).exists() and (d / n).stat().st_size >= s for n, s in FILES.items())


def runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def download_models(progress=lambda stage, done, total: True):
    """Скачивает оба файла (с повторами). progress -> False отменяет."""
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, min_size in FILES.items():
        dest = d / name
        if dest.exists() and dest.stat().st_size >= min_size:
            continue
        last = None
        for attempt in range(3):
            try:
                from . import netsafe
                import hashlib
                req = urllib.request.Request(REPO + name, headers={"User-Agent": "mlide-model-downloader"})
                tmp = dest.with_suffix(".part")
                h = hashlib.sha256()
                with netsafe.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        chunk = r.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if not progress(name, done, total):
                            tmp.unlink(missing_ok=True)
                            raise RuntimeError("Отменено")
                if tmp.stat().st_size < min_size or (total and tmp.stat().st_size != total):
                    raise IOError("файл скачан не полностью")
                if h.hexdigest() != SHA256[name]:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"Контрольная сумма файла {name} не совпала: файл повреждён или подменён")
                tmp.replace(dest)
                last = None
                break
            except RuntimeError:
                raise
            except Exception as e:  # noqa
                last = e
                try:
                    dest.with_suffix(".part").unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(2 * (attempt + 1))
        if last:
            from . import netsafe
            raise RuntimeError(f"Не удалось скачать {name}: {netsafe.explain(last)}")


class SamModel:
    """Интерактивный сегментатор. set_image() считает «отпечаток» картинки (1-3 с), predict() - мгновенно."""

    def __init__(self, threads: int = 4):
        import onnxruntime as ort
        d = models_dir()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / "mobile_sam_image_encoder.onnx"), opts, providers=prov)
        self.dec = ort.InferenceSession(str(d / "sam_mask_decoder_single.onnx"), opts, providers=prov)
        self.emb = None
        self.size = (0, 0)
        self.rsize = (0, 0)
        self.k = 1.0
        self.key = None

    def set_image(self, img_bgr, key=None):
        if key is not None and key == self.key and self.emb is not None:
            return
        h, w = img_bgr.shape[:2]
        self.k = 1024.0 / max(h, w)          # приводим длинную сторону к 1024: так кодировщик и декодер согласованы
        rh, rw = max(1, round(h * self.k)), max(1, round(w * self.k))
        small = cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_AREA if self.k < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.emb = self.enc.run(None, {"input_image": rgb})[0]
        self.size = (h, w)
        self.rsize = (rh, rw)
        self.key = key

    def predict(self, points, box=None, prev_mask=None):
        """points: [(x, y, label)] label 1 - объект, 0 - не объект. box: (x1,y1,x2,y2). Возвращает (маска uint8 0/255, iou)."""
        assert self.emb is not None, "set_image() не вызывался"
        h, w = self.size
        rh, rw = self.rsize
        k = self.k                                   # координаты переводятся в систему с длинной стороной 1024
        coords, labels = [], []
        for x, y, lb in points:
            coords.append([x * k, y * k])
            labels.append(float(lb))
        if box is not None:
            x1, y1, x2, y2 = box
            coords += [[x1 * k, y1 * k], [x2 * k, y2 * k]]
            labels += [2.0, 3.0]
        else:
            coords.append([0.0, 0.0])
            labels.append(-1.0)
        mask_in = np.zeros((1, 1, 256, 256), np.float32) if prev_mask is None else prev_mask
        out = self.dec.run(None, {
            "image_embeddings": self.emb,
            "point_coords": np.array([coords], np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": mask_in,
            "has_mask_input": np.array([0.0 if prev_mask is None else 1.0], np.float32),
            "orig_im_size": np.array([rh, rw], np.float32),
        })
        masks, ious = out[0], out[1]
        best = int(np.argmax(ious[0])) if ious.shape[1] > 1 else 0
        mask = (masks[0, best] > 0).astype(np.uint8) * 255
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask, float(ious[0][best])


def mask_to_polygon(mask, epsilon: float = 0.003, min_area_frac: float = 0.0005):
    """Маска -> полигон [[x,y],...] (самый большой контур, упрощённый). None, если пусто."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * mask.shape[0] * mask.shape[1]:
        return None
    c = cv2.approxPolyDP(c, epsilon * cv2.arcLength(c, True), True)
    if len(c) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in c]


def mask_to_box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
