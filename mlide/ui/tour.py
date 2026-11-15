"""Интерактивный тур: затемнение окна, подсветка нужного элемента и карточка с пояснением."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from . import icons, style
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def rect_of(win: QWidget, *widgets) -> QRect | None:
    """Объединённый прямоугольник виджетов в координатах окна."""
    r = None
    for w in widgets:
        if w is None or not w.isVisible():
            continue
        tl = w.mapTo(win, QPoint(0, 0))
        rr = QRect(tl, w.size())
        r = rr if r is None else r.united(rr)
    return r


class TourOverlay(QWidget):
    """steps: [(callable -> QRect|None, заголовок, html-текст)]"""

    def __init__(self, win: QWidget, steps, on_finish=None):
        super().__init__(win)
        self.win, self.steps, self.i, self.on_finish = win, steps, 0, on_finish
        self.hole: QRect | None = None
        self.setGeometry(win.rect())
        win.installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)

        self.card = QFrame(self)
        self.card.setObjectName("tourcard")
        self.card.setFixedWidth(400)
        self.card.setStyleSheet(
            f"#tourcard{{background:{style.PANEL};border:2px solid {style.ACCENT};border-radius:10px;}}"
            "QLabel{background:transparent;} QPushButton{padding:6px 14px;}")
        v = QVBoxLayout(self.card)
        v.setContentsMargins(16, 14, 16, 12)
        self.step_lbl = QLabel()
        self.step_lbl.setStyleSheet(f"color:{style.MUTED};font-size:11px;")
        self.title = QLabel()
        self.title.setStyleSheet("font-size:16px;font-weight:bold;")
        self.title.setWordWrap(True)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        v.addWidget(self.step_lbl)
        v.addWidget(self.title)
        v.addWidget(self.text)
        row = QHBoxLayout()
        self.skip = QPushButton("Пропустить тур")
        self.skip.clicked.connect(self.finish)
        self.back = QPushButton("Назад")
        self.back.setIcon(icons.icon("prev"))
        self.back.clicked.connect(lambda: self.go(-1))
        self.next = QPushButton("Далее")
        self.next.setIcon(icons.primary_icon("next"))
        self.next.setLayoutDirection(Qt.RightToLeft)     # значок справа от текста
        self.next.setDefault(True)
        self.next.clicked.connect(lambda: self.go(1))
        row.addWidget(self.skip)
        row.addStretch(1)
        row.addWidget(self.back)
        row.addWidget(self.next)
        v.addLayout(row)
        self.show()
        self.raise_()
        self.setFocus()
        self.show_step()

    # ---- навигация
    def go(self, d):
        j = self.i + d
        if j >= len(self.steps):
            self.finish()
            return
        self.i = max(0, j)
        self.show_step()

    def finish(self):
        self.win.removeEventFilter(self)
        self.hide()
        self.deleteLater()
        if self.on_finish:
            self.on_finish()

    def show_step(self):
        getter, title, text = self.steps[self.i]
        try:
            self.hole = getter() if getter else None
        except Exception:
            self.hole = None
        self.step_lbl.setText(f"Шаг {self.i + 1} из {len(self.steps)}")
        self.title.setText(title)
        self.text.setText(text)
        self.back.setEnabled(self.i > 0)
        last = self.i == len(self.steps) - 1
        self.next.setText("Закончить" if last else "Далее")
        self.next.setIcon(icons.primary_icon("accept" if last else "next"))
        self.skip.setVisible(not last)
        self.card.adjustSize()
        self._place_card()
        self.update()

    def _place_card(self):
        W, H = self.width(), self.height()
        cw, ch = self.card.width(), self.card.sizeHint().height()
        self.card.resize(cw, ch)
        if not self.hole:
            self.card.move((W - cw) // 2, (H - ch) // 2)
            return
        h = self.hole.adjusted(-8, -8, 8, 8)
        gap = 14
        spots = [
            QPoint(h.right() + gap, h.top()),                     # справа
            QPoint(h.left() - gap - cw, h.top()),                 # слева
            QPoint(h.left(), h.bottom() + gap),                   # снизу
            QPoint(h.left(), h.top() - gap - ch),                 # сверху
        ]
        for p in spots:
            if p.x() >= 8 and p.y() >= 8 and p.x() + cw <= W - 8 and p.y() + ch <= H - 8:
                self.card.move(p)
                return
        # не поместилось рядом: ставим в угол, наименее перекрывающий подсветку
        corners = [QPoint(W - cw - 16, H - ch - 16), QPoint(16, H - ch - 16), QPoint(W - cw - 16, 16), QPoint(16, 16)]
        best = min(corners, key=lambda p: QRect(p, self.card.size()).intersected(h).width() *
                   QRect(p, self.card.size()).intersected(h).height())
        self.card.move(best)

    # ---- рисование / события
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        if self.hole:
            hr = QRectF(self.hole.adjusted(-6, -6, 6, 6))
            hp = QPainterPath()
            hp.addRoundedRect(hr, 8, 8)
            path = path.subtracted(hp)
        p.fillPath(path, QColor(0, 0, 0, 165))
        if self.hole:
            p.setPen(QPen(QColor(style.ACCENT), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(self.hole.adjusted(-6, -6, 6, 6)), 8, 8)

    def eventFilter(self, obj, ev):
        if obj is self.win and ev.type() in (QEvent.Resize, QEvent.Move):
            self.setGeometry(self.win.rect())
            self.show_step()
        return False

    def mousePressEvent(self, e):
        e.accept()                  # блокируем клики по приложению под затемнением

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.finish()
        elif e.key() in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.go(1)
        elif e.key() == Qt.Key_Left:
            self.go(-1)
        else:
            e.accept()
