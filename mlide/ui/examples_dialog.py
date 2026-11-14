"""Окно «Примеры»: выбор учебного набора, скачивание и открытие как проекта."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
                               QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from .. import examples as EX
from .. import paths


class _Worker(QThread):
    progress = Signal(str, int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, ex):
        super().__init__()
        self.ex, self._cancel = ex, False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            p = EX.build_example(self.ex, lambda s, d, t: (self.progress.emit(s, d, t), not self._cancel)[1])
            self.done.emit(str(p))
        except Exception as e:  # noqa
            self.failed.emit(str(e))


class ExamplesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Примеры")
        self.resize(900, 560)
        self.result_path: str | None = None
        self.worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel("Учебные наборы с готовой разметкой: откройте любой и пройдите весь путь - от просмотра "
                       "разметки до обучения и результатов. Набор скачивается один раз с официального GitHub "
                       "(Ultralytics) в папку <code>" + str(paths.examples_dir()) + "</code>.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        sp = QSplitter()
        self.lst = QListWidget()
        for e in EX.EXAMPLES:
            ready = (paths.examples_dir() / e["id"] / f"{e['id']}.mlide.json").exists()
            it = QListWidgetItem(f"{e['title']}\n{e['kind']} · {len(e['names']) if not e.get('keep_top') else e['keep_top']} кл. · "
                                 f"~{e['size_mb']} МБ{'  ✔ скачан' if ready else ''}")
            it.setData(Qt.UserRole, e["id"])
            self.lst.addItem(it)
        self.lst.currentRowChanged.connect(self._show)
        sp.addWidget(self.lst)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.info = QTextBrowser()
        rv.addWidget(self.info, 1)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        rv.addWidget(self.bar)
        rv.addWidget(self.status)
        row = QHBoxLayout()
        self.go = QPushButton("Скачать и открыть")
        self.go.setProperty("primary", True)
        self.go.clicked.connect(self._run)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(close)
        row.addWidget(self.go)
        rv.addLayout(row)
        sp.addWidget(right)
        sp.setSizes([340, 560])
        lay.addWidget(sp, 1)
        self.lst.setCurrentRow(0)

    def current(self):
        it = self.lst.currentItem()
        return EX.BY_ID[it.data(Qt.UserRole)] if it else None

    def _show(self, _):
        e = self.current()
        if not e:
            return
        ready = (paths.examples_dir() / e["id"] / f"{e['id']}.mlide.json").exists()
        self.go.setText("Открыть" if ready else "Скачать и открыть")
        self.info.setHtml(
            f"<h2>{e['title']}</h2><p><b>{e['kind']}</b> · классы: {', '.join(e['names'][:6]) if not e.get('keep_top') else 'топ-' + str(e['keep_top']) + ' по частоте'}"
            f" · размер загрузки ~{e['size_mb']} МБ · уровень: {e['level']}</p>"
            f"<p>{e['what']}</p><p><b>Чему научит:</b> {e['learn']}</p>"
            f"<p style='color:#8b92a1'>Лицензия данных: {EX._LICENSE}</p>")

    def _run(self):
        e = self.current()
        if not e:
            return
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.worker = _Worker(e)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, stage, done, total):
        if total:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
            self.status.setText(f"{stage}: {done / 1048576:.1f} / {total / 1048576:.1f} МБ")
        else:
            self.bar.setRange(0, 0)
            self.status.setText(stage + "…")

    def _done(self, path):
        self.result_path = path
        self.accept()

    def _failed(self, msg):
        self.bar.setVisible(False)
        self.go.setEnabled(True)
        self.status.setText(f"<span style='color:#ff6b6b'>{msg}</span>")

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().reject()
