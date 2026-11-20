"""Стартовый экран: показывается, пока в проекте нет изображений."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import __version__, paths
from . import icons, style


class WelcomePage(QWidget):
    importImages = Signal()
    importFolder = Signal()
    openProject = Signal()
    openExamples = Signal()
    startTour = Signal()
    openRecent = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"WelcomePage {{ background: {style.CANVAS}; }}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        col = QVBoxLayout()
        col.setAlignment(Qt.AlignHCenter)

        head = QHBoxLayout()
        head.setAlignment(Qt.AlignHCenter)
        logo = QLabel()
        pm = QPixmap(str(paths.resource("assets/icon.png")))
        if not pm.isNull():
            logo.setPixmap(pm.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        head.addWidget(logo)
        t = QVBoxLayout()
        title = QLabel("ML IDE")
        title.setStyleSheet("font-size: 30px; font-weight: 700;")
        sub = QLabel(f"Разметка изображений и обучение YOLO · v{__version__}")
        sub.setStyleSheet(f"color: {style.MUTED}; font-size: 13px;")
        t.addWidget(title)
        t.addWidget(sub)
        head.addLayout(t)
        col.addLayout(head)
        col.addSpacing(26)

        grid = QGridLayout()
        grid.setSpacing(14)
        items = [
            ("images", "Импорт изображений", "Выберите файлы или перетащите их в окно", self.importImages, True),
            ("folder", "Импорт папки", "Все изображения из папки, включая вложенные", self.importFolder, False),
            ("open", "Открыть проект", "Продолжить работу или посмотреть результаты обучения", self.openProject, False),
            ("examples", "Примеры", "Готовые наборы с разметкой: скачать и попробовать", self.openExamples, False),
        ]
        for i, (ico, title_t, desc, sig, primary) in enumerate(items):
            grid.addWidget(self._card(ico, title_t, desc, sig, primary), i // 2, i % 2)
        col.addLayout(grid)
        self.recent_box = QVBoxLayout()
        self.recent_box.setAlignment(Qt.AlignHCenter)
        col.addLayout(self.recent_box)
        col.addSpacing(10)
        tour = QPushButton("Пройти тур по приложению")
        tour.setIcon(icons.icon("tour", style.ACCENT_HI))
        tour.setCursor(Qt.PointingHandCursor)
        tour.setFlat(True)
        tour.setStyleSheet(f"QPushButton{{color:{style.ACCENT_HI};border:none;font-size:13px;}} QPushButton:hover{{color:{style.TEXT};}}")
        tour.clicked.connect(self.startTour)
        col.addWidget(tour, 0, Qt.AlignHCenter)
        col.addSpacing(10)
        hint = QLabel("Путь: импорт → разметка → проверка → экспорт → обучение → результаты   ·   F1 - справочник")
        hint.setStyleSheet(f"color: {style.MUTED};")
        col.addWidget(hint, 0, Qt.AlignHCenter)
        outer.addLayout(col)
        outer.addStretch(1)

    def set_recent(self, items):
        while self.recent_box.count():
            w = self.recent_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not items:
            return
        lab = QLabel("Недавние проекты")
        lab.setStyleSheet(f"color:{style.MUTED};font-size:12px;")
        lab.setAlignment(Qt.AlignHCenter)
        self.recent_box.addWidget(lab)
        from pathlib import Path
        for p in items[:4]:
            b = QPushButton(Path(p).name)
            b.setIcon(icons.icon("file", style.ACCENT_HI))
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(p)
            b.setStyleSheet(f"QPushButton{{color:{style.ACCENT_HI};border:none;font-size:13px;padding:3px 10px;}} QPushButton:hover{{color:{style.TEXT};}}")
            b.clicked.connect(lambda _=False, p=p: self.openRecent.emit(p))
            self.recent_box.addWidget(b, 0, Qt.AlignHCenter)

    def _card(self, ico, title, desc, sig, primary):
        f = QFrame()
        f.setMinimumSize(330, 92)
        f.setMaximumWidth(330)
        bg = style.CARD_PRIMARY if primary else style.CARD
        f.setStyleSheet(f"QFrame{{background:{bg};border:1px solid {style.ACCENT if primary else style.BORDER};border-radius:12px;}}"
                        f"QFrame:hover{{border-color:{style.ACCENT_HI};}} QLabel{{background:transparent;border:none;}}")
        f.setCursor(Qt.PointingHandCursor)
        v = QVBoxLayout(f)
        v.setContentsMargins(16, 12, 16, 12)
        head = QHBoxLayout()
        il = QLabel()
        il.setPixmap(icons.pixmap(ico, style.ACCENT if not primary else style.ACCENT_HI, 22))
        head.addWidget(il)
        a = QLabel(title)
        a.setStyleSheet("font-size: 15px; font-weight: 600;")
        head.addWidget(a, 1)
        b = QLabel(desc)
        b.setStyleSheet(f"color: {style.MUTED};")
        b.setWordWrap(True)
        v.addLayout(head)
        v.addWidget(b)
        f.mousePressEvent = lambda e, s=sig: s.emit()
        return f
