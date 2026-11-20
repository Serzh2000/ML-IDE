"""Unicode-safe image I/O (cv2.imread breaks on non-ASCII Windows paths)."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(p) -> bool:
    return Path(p).suffix.lower() in IMAGE_EXTS


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    try:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # applies EXIF orientation
    except cv2.error:                                # пустой или битый файл: cv2.error вместо None
        img = None
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def imwrite(path, img, quality: int = 95):
    ext = Path(path).suffix.lower() or ".jpg"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def exif_orientation(path) -> int:
    try:
        with Image.open(path) as im:
            return int(im.getexif().get(274, 1))
    except Exception:
        return 1


def size_of(path) -> tuple[int, int]:
    """(width, height) honouring EXIF rotation, same as imread()."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            if im.format != "TIFF" and int(im.getexif().get(274, 1)) in (5, 6, 7, 8):      # TIFF: Pillow уже поменял размер местами
                w, h = h, w
        except Exception:
            pass
    return w, h


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)
