"""Поведение окон приложения, общее для всех диалогов:
- кнопка «развернуть» и уголок для растяжения (в Windows у диалогов их по умолчанию нет);
- окно не больше экрана: если содержимое не помещается (маленький ноутбук, масштаб 125-150%), оно оборачивается в прокрутку;
- потоки QThread не уничтожаются на ходу: закрытие окна во время работы потока раньше обрывало процесс («Destroyed while thread is running»).
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, QThread
from PySide6.QtWidgets import QApplication, QDialog, QFrame, QScrollArea, QVBoxLayout, QWidget

_RUNNING: set = set()                         # запущенные потоки: ссылка живёт, пока поток не завершится


def available_rect(widget=None):
    scr = None
    if widget is not None and widget.windowHandle() is not None:
        scr = widget.windowHandle().screen()
    scr = scr or (widget.screen() if widget is not None else None) or QApplication.primaryScreen()
    return scr.availableGeometry() if scr else None


def scroll_layout(dlg: QWidget, layout_cls=QVBoxLayout):
    """Корневой layout окна внутри области прокрутки: на маленьком экране окно можно сузить, а содержимое прокручивается."""
    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(0, 0, 0, 0)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    body = QWidget()
    area.setWidget(body)
    outer.addWidget(area)
    return layout_cls(body)


def fit_to_screen(dlg: QWidget, avail=None, ratio: float = 0.94) -> bool:
    """Подгоняет размер окна под экран. -> True, если размер пришлось уменьшить."""
    avail = avail or available_rect(dlg)
    if avail is None:
        return False
    max_w, max_h = int(avail.width() * ratio), int(avail.height() * ratio)
    w, h = dlg.width(), dlg.height()
    if w > max_w or h > max_h:
        dlg.resize(min(w, max_w), min(h, max_h))
        if dlg.isVisible():
            geo = dlg.frameGeometry()
            dlg.move(max(avail.left(), min(geo.left(), avail.right() - geo.width())), max(avail.top(), min(geo.top(), avail.bottom() - geo.height())))
    return w > max_w or h > max_h


class _ShowFilter(QObject):
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Show and not getattr(obj, "_mlide_fit_done", False):
            obj._mlide_fit_done = True
            try:
                fit_to_screen(obj)
            except RuntimeError:
                pass
        return False


_FILTER: _ShowFilter | None = None


def install():
    global _FILTER
    if getattr(QDialog, "_mlide_flags", False):
        return
    _FILTER = _ShowFilter()
    orig = QDialog.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        if type(self).__module__.startswith("mlide"):
            self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
            self.setSizeGripEnabled(True)
            self.installEventFilter(_FILTER)

    QDialog.__init__ = init
    QDialog._mlide_flags = True

    _install_recolor()
    start = QThread.start

    def tstart(self, *a):
        _RUNNING.add(self)
        self.finished.connect(lambda t=self: _RUNNING.discard(t))
        return start(self, *a)

    QThread.start = tstart


def wait_threads(ms: int = 2500):
    """При выходе из приложения даём потокам завершиться (они просят остановиться через requestInterruption)."""
    for t in list(_RUNNING):
        try:
            t.requestInterruption()
            t.quit()
            t.wait(ms)
        except RuntimeError:
            pass


def _install_recolor():
    """Светлая тема: цвета в setText/setHtml/setStyleSheet и конструкторах подставляются через style.recolor (см. LIGHT_FIX)."""
    from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTextBrowser, QTextEdit, QWidget
    from . import style

    def wrap(cls, name):
        orig = getattr(cls, name)

        def f(self, text, *a, **k):
            return orig(self, style.recolor(text) if isinstance(text, str) else text, *a, **k)
        setattr(cls, name, f)

    for cls, names in ((QLabel, ("setText",)), (QTextEdit, ("setHtml", "setText")), (QTextBrowser, ("setHtml", "setText")),
                       (QWidget, ("setStyleSheet", "setToolTip"))):
        for n in names:
            try:
                wrap(cls, n)
            except Exception:  # noqa
                pass
    init = QLabel.__init__

    def label_init(self, *a, **k):
        a = tuple(style.recolor(x) if isinstance(x, str) else x for x in a)
        init(self, *a, **k)
    QLabel.__init__ = label_init
