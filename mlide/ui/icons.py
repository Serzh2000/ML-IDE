"""Иконки интерфейса: векторный набор Lucide (ISC, assets/icons), окрашиваются в цвета текущей темы."""
from __future__ import annotations

import base64
from functools import lru_cache

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .. import paths
from . import style


ALIAS = {"poly": "polygon"}          # ключи инструментов -> имена файлов значков


@lru_cache(maxsize=None)
def _svg(name: str) -> str:
    name = ALIAS.get(name, name)
    p = paths.resource(f"assets/icons/{name}.svg")
    if not p.exists():
        raise FileNotFoundError(f"нет значка {name}")
    return p.read_text(encoding="utf-8")


def pixmap(name: str, color: str | None = None, size: int = 18, dpr: float = 2.0, stroke: float | None = None) -> QPixmap:
    color = color or style.TEXT
    svg = _svg(name).replace("currentColor", color)
    if stroke is not None:
        svg = svg.replace('stroke-width="2"', f'stroke-width="{stroke}"')
    r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p)
    p.end()
    px.setDevicePixelRatio(dpr)
    return px


def icon(name: str, color: str | None = None, size: int = 18, checked_color: str = "#ffffff") -> QIcon:
    """Значок для кнопок и панелей: обычный, выбранный (белый на акценте) и недоступный."""
    ic = QIcon()
    normal = color or style.TEXT
    ic.addPixmap(pixmap(name, normal, size), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(name, normal, size), QIcon.Active, QIcon.Off)
    ic.addPixmap(pixmap(name, checked_color, size), QIcon.Normal, QIcon.On)
    ic.addPixmap(pixmap(name, checked_color, size), QIcon.Active, QIcon.On)
    ic.addPixmap(pixmap(name, style.DISABLED, size), QIcon.Disabled, QIcon.Off)
    ic.addPixmap(pixmap(name, style.DISABLED, size), QIcon.Disabled, QIcon.On)
    return ic


def primary_icon(name: str, size: int = 18) -> QIcon:
    """Значок на акцентной (синей) кнопке: всегда белый."""
    return icon(name, "#ffffff", size)


def tinted(name: str, color: str, size: int = 16) -> QIcon:
    return QIcon(pixmap(name, color, size))


def data_uri(name: str, color: str, size: int = 16) -> str:
    """PNG в base64 для вставки в HTML (<img src=...>)."""
    px = pixmap(name, color, size, dpr=2.0)
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    px.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(bytes(buf.data())).decode("ascii")


STATUS_COLOR = {"ok": "#3fb950", "warn": "#f0a030", "info": "#58a6ff"}


def status_img(level: str, size: int = 16) -> str:
    """HTML-тег значка статуса (галочка / предупреждение / информация) вместо эмодзи."""
    return f'<img src="{data_uri(level, STATUS_COLOR[level], size)}" width="{size}" height="{size}" style="vertical-align:middle">'
