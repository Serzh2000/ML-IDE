"""Генерирует иконку приложения: python tools/make_icon.py  ->  assets/icon.png, assets/icon.ico"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)


def render(size=512) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size
    # фон: скруглённый квадрат с градиентом
    g = QLinearGradient(0, 0, s, s)
    g.setColorAt(0, QColor("#4f7cff"))
    g.setColorAt(1, QColor("#8a5cff"))
    path = QPainterPath()
    path.addRoundedRect(QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92), s * 0.22, s * 0.22)
    p.fillPath(path, g)
    # рамка разметки с угловыми скобками
    pen = QPen(QColor("white"), s * 0.045, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    m, L = s * 0.24, s * 0.16
    x0, y0, x1, y1 = m, m, s - m, s - m
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        p.drawLine(QPointF(cx, cy), QPointF(cx + dx * L, cy))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * L))
    # объект внутри: «цель»
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 235))
    p.drawEllipse(QPointF(s / 2, s / 2), s * 0.10, s * 0.10)
    p.setBrush(QColor("#6a63ff"))
    p.drawEllipse(QPointF(s / 2, s / 2), s * 0.045, s * 0.045)
    p.end()
    return img


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    img = render(512)
    png = OUT / "icon.png"
    img.save(str(png))
    Image.open(png).save(OUT / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("saved", png, OUT / "icon.ico")
