"""Нарезка видео на кадры для разметки."""
from __future__ import annotations

from . import winflags

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout)

from .. import paths, video
from .dialogs import add_tips


class _Worker(QThread):
    progress = Signal(int, int)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, vpath, out_dir, kw):
        super().__init__()
        self.vpath, self.out_dir, self.kw, self._cancel = vpath, out_dir, kw, False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = video.extract_frames(self.vpath, self.out_dir, progress=lambda d, t: (self.progress.emit(d, t), not self._cancel)[1],
                                       **self.kw)
            self.done.emit([str(p) for p in res])
        except Exception as e:  # noqa
            self.failed.emit(str(e))


class VideoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Кадры из видео")
        self.resize(640, 420)
        self.result_paths: list[str] = []
        self.add_to_project = True
        self.worker: _Worker | None = None
        lay = winflags.scroll_layout(self)
        self._root = lay
        intro = QLabel("Нарезает видео на отдельные кадры для разметки. Совет: берите не все кадры, а один в 0.5-2 секунды "
                       "или только при смене сцены: соседние кадры почти одинаковы и не добавляют знаний модели.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        row = QHBoxLayout()
        self.video = QLineEdit()
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        row.addWidget(self.video, 1)
        row.addWidget(b)
        f.addRow("Видеофайл", row)
        self.info = QLabel("")
        f.addRow("", self.info)
        self.mode = QComboBox()
        self.mode.addItems(["Каждые N секунд", "Каждый N-й кадр", "При смене сцены"])
        self.mode.currentIndexChanged.connect(self._mode_changed)
        f.addRow("Режим", self.mode)
        self.val_sec = QDoubleSpinBox()
        self.val_sec.setRange(0.05, 3600)
        self.val_sec.setValue(1.0)
        self.val_sec.setSuffix(" с")
        self.val_n = QSpinBox()
        self.val_n.setRange(1, 100000)
        self.val_n.setValue(15)
        self.val_scene = QDoubleSpinBox()
        self.val_scene.setRange(0.01, 0.9)
        self.val_scene.setSingleStep(0.01)
        self.val_scene.setDecimals(2)
        self.val_scene.setValue(0.08)
        for w in (self.val_sec, self.val_n, self.val_scene):
            f.addRow("Значение", w)
        self.maxf = QSpinBox()
        self.maxf.setRange(0, 100000)
        self.maxf.setSpecialValueText("без ограничения")
        f.addRow("Максимум кадров", self.maxf)
        row = QHBoxLayout()
        self.out = QLineEdit()
        b2 = QPushButton("Обзор…")
        b2.clicked.connect(self._browse_out)
        row.addWidget(self.out, 1)
        row.addWidget(b2)
        f.addRow("Папка для кадров", row)
        self.add = QCheckBox("Сразу добавить кадры в проект")
        self.add.setChecked(True)
        f.addRow("", self.add)
        lay.addLayout(f)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        self.go = QPushButton("Извлечь кадры")
        self.go.setProperty("primary", True)
        self.go.clicked.connect(self._run)
        row.addWidget(close)
        row.addWidget(self.go)
        lay.addLayout(row)
        add_tips(self, {
            self.mode: "Как выбирать кадры: через равные промежутки времени, каждый N-й или только когда картинка заметно меняется.",
            self.val_sec: "Один кадр в указанное число секунд. Для медленной сцены 1-2 с, для быстрой 0.2-0.5 с.",
            self.val_n: "Брать каждый N-й кадр видео (при 30 кадр/с значение 15 даёт 2 кадра в секунду).",
            self.val_scene: "Чувствительность к смене сцены: 0.05 - ловит небольшие изменения, 0.10-0.15 - только резкие смены.",
            self.maxf: "Остановиться после указанного числа кадров (0 - без ограничения).",
            self.out: "Куда сохранить кадры (JPEG). По умолчанию - папка «ML IDE/frames» в вашей домашней папке.",
            self.add: "После извлечения кадры автоматически появятся в списке изображений проекта.",
        })
        self._mode_changed()

    def _mode_changed(self):
        i = self.mode.currentIndex()
        for k, w in enumerate((self.val_sec, self.val_n, self.val_scene)):
            w.setVisible(k == i)
            lab = self._root.itemAt(1).layout().labelForField(w)
            if lab:
                lab.setVisible(k == i)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Видео", "", "Видео (*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v);;Все файлы (*)")
        if p:
            self.video.setText(p)
            try:
                i = video.video_info(p)
                self.info.setText(f"{i['width']}×{i['height']}, {i['fps']:.1f} кадр/с, {i['duration_sec']:.1f} с, ≈{i['frames']} кадров")
            except Exception as e:  # noqa
                self.info.setText(f"<span style='color:#ff6b6b'>{e}</span>")
            if not self.out.text():
                self.out.setText(str(paths.examples_dir().parent / "frames" / Path(p).stem))

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Папка для кадров", self.out.text())
        if d:
            self.out.setText(d)

    def _run(self):
        v = self.video.text().strip()
        if not v or not Path(v).exists():
            QMessageBox.warning(self, "Кадры из видео", "Выберите видеофайл.")
            return
        out = self.out.text().strip() or str(paths.examples_dir().parent / "frames" / Path(v).stem)
        kw = {"max_frames": self.maxf.value() or None}
        i = self.mode.currentIndex()
        if i == 0:
            kw["every_sec"] = self.val_sec.value()
        elif i == 1:
            kw["every_n"] = self.val_n.value()
        else:
            kw["scene_threshold"] = self.val_scene.value()
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.worker = _Worker(v, out, kw)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, d, t):
        if t:
            self.bar.setRange(0, t)
            self.bar.setValue(d)
        self.status.setText(f"Обработано кадров видео: {d}" + (f" из {t}" if t else ""))

    def _done(self, paths_):
        self.result_paths = paths_
        self.add_to_project = self.add.isChecked()
        self.status.setText(f"Готово: сохранено кадров - {len(paths_)}.")
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
