"""Augmentation presets + offline augmentation that keeps boxes/polygons in sync."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, fields

import cv2
import numpy as np


@dataclass
class AugParams:
    # geometry (same meaning as Ultralytics hyper-parameters)
    fliplr: float = 0.5        # probability
    flipud: float = 0.0        # probability
    degrees: float = 0.0       # +-deg
    translate: float = 0.1     # +-fraction of image size
    scale: float = 0.5         # factor in [1-s, 1+s]
    shear: float = 0.0         # +-deg
    perspective: float = 0.0   # online only
    # colour
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    brightness: float = 0.0    # +-fraction (offline only)
    contrast: float = 0.0      # +-fraction (offline only)
    # extras (offline only; probabilities)
    blur: float = 0.0
    noise: float = 0.0
    grayscale: float = 0.0
    cutout: float = 0.0
    # online-only (Ultralytics trainer)
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    # offline expansion
    copies: int = 2            # augmented copies per training image

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        names = {f.name for f in fields(AugParams)}
        return AugParams(**{k: v for k, v in (d or {}).items() if k in names})

    def ultralytics_hyp(self) -> dict:
        """Hyper-parameters understood by `model.train(**hyp)`."""
        return {
            "hsv_h": self.hsv_h, "hsv_s": self.hsv_s, "hsv_v": self.hsv_v,
            "degrees": self.degrees, "translate": self.translate, "scale": self.scale,
            "shear": self.shear, "perspective": self.perspective,
            "flipud": self.flipud, "fliplr": self.fliplr,
            "mosaic": self.mosaic, "mixup": self.mixup, "copy_paste": self.copy_paste,
            "erasing": self.cutout,
        }


# description shown in UI for every field: (label, min, max, step, decimals, tooltip)
FIELD_UI = {
    "fliplr": ("Отражение по горизонтали (вер.)", 0, 1, 0.05, 2, "Вероятность зеркального отражения слева-направо"),
    "flipud": ("Отражение по вертикали (вер.)", 0, 1, 0.05, 2, "Вероятность отражения сверху-вниз (аэросъёмка, микроскоп: да; дорожные сцены: нет)"),
    "degrees": ("Поворот (± градусов)", 0, 180, 1, 1, "Диапазон случайного поворота"),
    "translate": ("Сдвиг (± доля)", 0, 0.9, 0.01, 2, "Случайный сдвиг как доля размера изображения"),
    "scale": ("Масштаб (± доля)", 0, 0.95, 0.05, 2, "Диапазон приближения/отдаления: 0.5 = от 50% до 150%"),
    "shear": ("Скос (± градусов)", 0, 45, 1, 1, "Диапазон случайного скоса"),
    "perspective": ("Перспектива (онлайн)", 0, 0.001, 0.0001, 4, "Только онлайн-аугментация Ultralytics"),
    "hsv_h": ("Сдвиг оттенка", 0, 0.5, 0.005, 3, "Изменение оттенка цвета"),
    "hsv_s": ("Насыщенность", 0, 1, 0.05, 2, "Диапазон изменения насыщенности"),
    "hsv_v": ("Яркость (HSV)", 0, 1, 0.05, 2, "Диапазон изменения яркости"),
    "brightness": ("Яркость (± доля)", 0, 1, 0.05, 2, "Только офлайн: добавочная яркость"),
    "contrast": ("Контраст (± доля)", 0, 1, 0.05, 2, "Только офлайн: коэффициент контраста"),
    "blur": ("Размытие по Гауссу (вер.)", 0, 1, 0.05, 2, "Только офлайн: имитация расфокуса / смаза"),
    "noise": ("Шум (вер.)", 0, 1, 0.05, 2, "Только офлайн: шум матрицы камеры"),
    "grayscale": ("Чёрно-белое (вер.)", 0, 1, 0.05, 2, "Только офлайн"),
    "cutout": ("Вырезы / стирание (вер.)", 0, 1, 0.05, 2, "Случайные серые прямоугольники; делает модель устойчивой к перекрытиям"),
    "mosaic": ("Мозаика (онлайн)", 0, 1, 0.05, 2, "Только онлайн Ultralytics: 4 изображения склеиваются в одно"),
    "mixup": ("MixUp (онлайн)", 0, 1, 0.05, 2, "Только онлайн Ultralytics"),
    "copy_paste": ("Copy-paste (онлайн)", 0, 1, 0.05, 2, "Только онлайн Ultralytics, для сегментации"),
    "copies": ("Офлайн-копий на изображение", 0, 50, 1, 0, "Сколько аугментированных копий добавить в обучающую выборку"),
}

BUILTIN_PRESETS: dict[str, AugParams] = {
    "Без аугментаций": AugParams(fliplr=0, flipud=0, translate=0, scale=0, hsv_h=0, hsv_s=0, hsv_v=0,
                      mosaic=0, copies=0),
    "Лёгкая": AugParams(fliplr=0.5, translate=0.05, scale=0.2, degrees=3, hsv_h=0.01, hsv_s=0.4,
                       hsv_v=0.25, mosaic=0.5, copies=1),
    "Средняя (по умолчанию YOLO)": AugParams(),
    "Сильная": AugParams(fliplr=0.5, flipud=0.1, degrees=15, translate=0.15, scale=0.6, shear=5,
                       hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, brightness=0.2, contrast=0.2, blur=0.2,
                       noise=0.2, cutout=0.3, mosaic=1.0, mixup=0.15, copies=4),
    "Аэросъёмка / дрон": AugParams(fliplr=0.5, flipud=0.5, degrees=180, translate=0.1, scale=0.4,
                                hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, blur=0.1, mosaic=1.0, copies=3),
    "Тёмное время / ночь": AugParams(fliplr=0.5, translate=0.1, scale=0.4, hsv_s=0.5, hsv_v=0.6,
                                   brightness=0.35, contrast=0.3, noise=0.5, blur=0.2,
                                   mosaic=0.8, copies=3),
}


def _color_jitter(img, p: AugParams, rng):
    if p.hsv_h or p.hsv_s or p.hsv_v:
        r = rng.uniform(-1, 1, 3) * [p.hsv_h, p.hsv_s, p.hsv_v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(256, dtype=np.float32)
        lut_h = ((x * r[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v))),
                           cv2.COLOR_HSV2BGR)
    if p.brightness or p.contrast:
        alpha = 1 + rng.uniform(-p.contrast, p.contrast)
        beta = rng.uniform(-p.brightness, p.brightness) * 255
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def augment_sample(img, anns, p: AugParams, rng: np.random.Generator, flip_idx=None):
    """img: BGR uint8. anns: list of (cls, kind, pts) или (cls, "pose", box_pts, kps[[x,y,v],...]). Returns (img, anns).

    Для pose отражение по горизонтали возможно только при заданном flip_idx (перестановка левых/правых точек);
    по вертикали поза не отражается.
    """
    h, w = img.shape[:2]
    has_pose = any(len(a) > 3 and a[1] == "pose" for a in anns)

    # --- geometric: one combined affine matrix (about image centre)
    flip_x = -1.0 if rng.random() < p.fliplr and (not has_pose or flip_idx) else 1.0
    flip_y = -1.0 if rng.random() < p.flipud and not has_pose else 1.0
    ang = math.radians(rng.uniform(-p.degrees, p.degrees))
    s = rng.uniform(1 - p.scale, 1 + p.scale) if p.scale else 1.0
    shx = math.tan(math.radians(rng.uniform(-p.shear, p.shear))) if p.shear else 0.0
    shy = math.tan(math.radians(rng.uniform(-p.shear, p.shear))) if p.shear else 0.0
    tx = rng.uniform(-p.translate, p.translate) * w
    ty = rng.uniform(-p.translate, p.translate) * h

    C = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], dtype=np.float64)
    F = np.diag([flip_x, flip_y, 1.0])
    R = np.array([[math.cos(ang) * s, -math.sin(ang) * s, 0],
                  [math.sin(ang) * s, math.cos(ang) * s, 0], [0, 0, 1]])
    S = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])
    T = np.array([[1, 0, w / 2 + tx], [0, 1, h / 2 + ty], [0, 0, 1]])
    M = T @ S @ R @ F @ C

    identity = np.allclose(M, np.eye(3))
    # разметка живёт в непрерывных координатах (кадр от 0 до w), а warpAffine работает по индексам пикселей (центр пикселя = целое):
    # без поправки в полпикселя метка после отражения смещалась на 1 px относительно картинки
    D = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64)
    Mi = np.linalg.inv(D) @ M @ D
    out = img if identity else cv2.warpAffine(img, Mi[:2], (w, h), flags=cv2.INTER_LINEAR,
                                              borderValue=(114, 114, 114))

    new_anns = []
    for item in anns:
        cls, kind, pts = item[:3]
        kps = item[3] if len(item) > 3 else None
        P = np.asarray(pts, dtype=np.float64)
        if kind in ("box", "pose"):
            x1, y1 = P.min(0)
            x2, y2 = P.max(0)
            P = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        Q = (np.c_[P, np.ones(len(P))] @ M.T)[:, :2]
        area_before = max(1e-6, np.ptp(Q[:, 0]) * np.ptp(Q[:, 1]))
        Q[:, 0] = Q[:, 0].clip(0, w)
        Q[:, 1] = Q[:, 1].clip(0, h)
        bw, bh = np.ptp(Q[:, 0]), np.ptp(Q[:, 1])
        if bw < 2 or bh < 2 or bw * bh < 0.25 * area_before:
            continue
        if kind == "pose":
            K = np.asarray(kps, dtype=np.float64).reshape(-1, 3)
            XY = (np.c_[K[:, :2], np.ones(len(K))] @ M.T)[:, :2]
            V = K[:, 2].copy()
            if flip_x < 0 and flip_idx:
                XY, V = XY[list(flip_idx)], V[list(flip_idx)]
            inside = (XY[:, 0] >= 0) & (XY[:, 0] <= w) & (XY[:, 1] >= 0) & (XY[:, 1] <= h)
            V = np.where(inside, V, 0)
            XY = np.where(V[:, None] > 0, XY, 0)
            new_anns.append((cls, "pose", [[Q[:, 0].min(), Q[:, 1].min()], [Q[:, 0].max(), Q[:, 1].max()]],
                             [[float(x), float(y), int(v)] for (x, y), v in zip(XY, V)]))
        elif kind == "box":
            new_anns.append((cls, "box", [[Q[:, 0].min(), Q[:, 1].min()], [Q[:, 0].max(), Q[:, 1].max()]]))
        elif kind == "obb":
            new_anns.append((cls, "obb", Q.tolist()))
        else:
            new_anns.append((cls, "poly", Q.tolist()))

    # --- photometric / noise
    out = _color_jitter(out, p, rng)
    if p.grayscale and rng.random() < p.grayscale:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if p.blur and rng.random() < p.blur:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if p.noise and rng.random() < p.noise:
        sigma = rng.uniform(4, 14)
        out = np.clip(out.astype(np.float32) + rng.normal(0, sigma, out.shape), 0, 255).astype(np.uint8)
    if p.cutout and rng.random() < p.cutout:
        out = out.copy()
        for _ in range(int(rng.integers(1, 4))):
            cw, ch = int(w * rng.uniform(0.05, 0.2)), int(h * rng.uniform(0.05, 0.2))
            x0, y0 = int(rng.integers(0, max(1, w - cw))), int(rng.integers(0, max(1, h - ch)))
            out[y0:y0 + ch, x0:x0 + cw] = int(rng.integers(90, 160))
    return out, new_anns


def get_all_presets(project_presets: dict | None = None) -> dict[str, AugParams]:
    d = dict(BUILTIN_PRESETS)
    for k, v in (project_presets or {}).items():
        d[k] = AugParams.from_dict(v)
    return d
