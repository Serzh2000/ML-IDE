"""Поиск дублей и очень похожих кадров."""
from __future__ import annotations

from . import winflags

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from .. import dedupe
from .dialogs import add_tips


class _Worker(QThread):
    progress = Signal(int, int)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, paths_, thr):
        super().__init__()
        self.paths_, self.thr, self._cancel = paths_, thr, False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            g = dedupe.find_near_duplicates(self.paths_, self.thr, lambda d, t: (self.progress.emit(d, t), not self._cancel)[1])
            self.done.emit(g)
        except Exception as e:  # noqa
            self.failed.emit(str(e))


class DedupeDialog(QDialog):
    goto = Signal(int)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Дубли и похожие кадры")
        self.resize(760, 560)
        self.project = project
        self.groups: list[list[str]] = []
        self.worker = None
        self.changed = False
        lay = winflags.scroll_layout(self)
        intro = QLabel("Находит одинаковые и почти одинаковые картинки (например, соседние кадры видео). Лишние копии "
                       "раздувают датасет и дают завышенные метрики: одна копия попадает в обучение, другая - в проверку.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Строгость:"))
        self.thr = QSpinBox()
        self.thr.setRange(0, 20)
        self.thr.setValue(6)
        row.addWidget(self.thr)
        row.addWidget(QLabel("(0 - только точные копии, 6 - стандарт, 12 - очень похожие)"))
        row.addStretch(1)
        self.go = QPushButton("Найти")
        self.go.setProperty("primary", True)
        self.go.clicked.connect(self._run)
        row.addWidget(self.go)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = QLabel("")
        lay.addWidget(self.status)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Файл", "Объектов"])
        self.tree.setColumnWidth(0, 520)
        self.tree.itemDoubleClicked.connect(self._open)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.keep = QPushButton("Оставить по одному в каждой группе")
        self.keep.setToolTip("<p>Из проекта удаляются все копии группы, кроме одной: остаётся та, где больше разметки. "
                             "Файлы на диске не трогаются.</p>")
        self.keep.setEnabled(False)
        self.keep.clicked.connect(self._keep_one)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        row.addWidget(self.keep)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        add_tips(self, {self.thr: "Насколько картинки могут отличаться, чтобы считаться похожими. Больше значение - больше групп, но и больше случайных совпадений."})

    def _entry(self, path):
        return next((e for e in self.project.images if e.path == path), None)

    def _run(self):
        if len(self.project.images) < 2:
            return
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.worker = _Worker([e.path for e in self.project.images], self.thr.value())
        self.worker.progress.connect(lambda d, t: (self.bar.setRange(0, t), self.bar.setValue(d)))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(lambda m: (self.status.setText(m), self.go.setEnabled(True)))
        self.worker.start()

    def _done(self, groups):
        self.groups = groups
        self.go.setEnabled(True)
        self.bar.setVisible(False)
        self.tree.clear()
        extra = 0
        for i, g in enumerate(groups, 1):
            top = QTreeWidgetItem([f"Группа {i} - {len(g)} шт.", ""])
            for p in g:
                e = self._entry(p)
                n = len([a for a in e.anns if not a.suggested]) if e else 0
                child = QTreeWidgetItem([Path(p).name, str(n)])
                child.setData(0, Qt.UserRole, p)
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            top.setExpanded(len(groups) <= 20)
            extra += len(g) - 1
        self.keep.setEnabled(bool(groups))
        self.status.setText(f"Найдено групп: {len(groups)}, лишних копий: {extra}." if groups else "Дублей не найдено")

    def _open(self, item, _c):
        p = item.data(0, Qt.UserRole)
        for i, e in enumerate(self.project.images):
            if e.path == p:
                self.goto.emit(i)
                break

    def _keep_one(self):
        drop = set()
        for g in self.groups:
            ents = [self._entry(p) for p in g if self._entry(p)]
            if not ents:
                continue
            best = max(ents, key=lambda e: len([a for a in e.anns if not a.suggested]))
            drop |= {e.path for e in ents if e is not best}
        if not drop:
            return
        if QMessageBox.question(self, "Удаление копий", f"Убрать из проекта {len(drop)} лишних изображений вместе с их разметкой?\n"
                                "Сами файлы на диске останутся.") != QMessageBox.Yes:
            return
        self.project.images = [e for e in self.project.images if e.path not in drop]
        self.project.dirty = True
        self.changed = True
        self.groups = []
        self.tree.clear()
        self.keep.setEnabled(False)
        self.status.setText(f"Убрано из проекта: {len(drop)}.")

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().reject()
