"""Стартовый экран: показывается, пока в проекте нет изображений."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import __version__, paths


class WelcomePage(QWidget):
    importImages = Signal()
    importFolder = Signal()
    openProject = Signal()
    openExamples = Signal()
    startTour = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("WelcomePage { background: #17191e; }")
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
        sub.setStyleSheet("color: #8b92a1; font-size: 13px;")
        t.addWidget(title)
        t.addWidget(sub)
        head.addLayout(t)
        col.addLayout(head)
        col.addSpacing(26)

        grid = QGridLayout()
        grid.setSpacing(14)
        items = [
            ("🖼  Импорт изображений", "Выберите файлы или перетащите их в окно", self.importImages, True),
            ("📁  Импорт папки", "Все изображения из папки, включая вложенные", self.importFolder, False),
            ("📂  Открыть проект", "Продолжить работу или посмотреть результаты обучения", self.openProject, False),
            ("🎓  Примеры", "Готовые наборы с разметкой: скачать и попробовать", self.openExamples, False),
        ]
        for i, (title_t, desc, sig, primary) in enumerate(items):
            grid.addWidget(self._card(title_t, desc, sig, primary), i // 2, i % 2)
        col.addLayout(grid)
        col.addSpacing(18)
        tour = QPushButton("▶  Пройти тур по приложению")
        tour.setCursor(Qt.PointingHandCursor)
        tour.setFlat(True)
        tour.setStyleSheet("QPushButton{color:#7aa2ff;border:none;font-size:13px;} QPushButton:hover{color:white;}")
        tour.clicked.connect(self.startTour)
        col.addWidget(tour, 0, Qt.AlignHCenter)
        col.addSpacing(10)
        hint = QLabel("Путь: импорт → разметка → проверка → экспорт → обучение → результаты   ·   F1 - справочник")
        hint.setStyleSheet("color: #6b7280;")
        col.addWidget(hint, 0, Qt.AlignHCenter)
        outer.addLayout(col)
        outer.addStretch(1)

    def _card(self, title, desc, sig, primary):
        f = QFrame()
        f.setFixedSize(330, 92)
        bg = "#2a3a66" if primary else "#232730"
        f.setStyleSheet(f"QFrame{{background:{bg};border:1px solid {'#5b8cff' if primary else '#363b47'};border-radius:12px;}}"
                        "QFrame:hover{border-color:#7aa2ff;} QLabel{background:transparent;border:none;}")
        f.setCursor(Qt.PointingHandCursor)
        v = QVBoxLayout(f)
        v.setContentsMargins(16, 12, 16, 12)
        a = QLabel(title)
        a.setStyleSheet("font-size: 15px; font-weight: 600;")
        b = QLabel(desc)
        b.setStyleSheet("color: #8b92a1;")
        b.setWordWrap(True)
        v.addWidget(a)
        v.addWidget(b)
        f.mousePressEvent = lambda e, s=sig: s.emit()
        return f
