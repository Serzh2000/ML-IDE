"""Окно «Экспорт модели»: .pt -> ONNX / TorchScript / OpenVINO / TensorRT / NCNN / TFLite / CoreML."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTextBrowser,
                               QVBoxLayout)

from .. import paths
from .. import modelexport as ME
from .dialogs import add_tips
from .learn import cards_html


class ModelExportDialog(QDialog):
    def __init__(self, project, parent=None, weights: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Экспорт обученной модели")
        self.resize(900, 760)
        self.project = project
        self.proc: QProcess | None = None
        self.result_path: Path | None = None
        self._log = ""

        lay = QVBoxLayout(self)
        intro = QLabel("Преобразует обученную модель <code>best.pt</code> в формат для вашего устройства или "
                       "приложения. Нужны Python с пакетом <code>ultralytics</code> (см. окно «Обучение»).")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        f = QFormLayout()
        row = QHBoxLayout()
        self.weights = QLineEdit(weights or self._guess_weights())
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        row.addWidget(self.weights, 1)
        row.addWidget(b)
        f.addRow("Веса модели (.pt)", row)
        self.py = QLineEdit(shutil.which("python") or shutil.which("py") or "")
        f.addRow("Python", self.py)
        self.fmt = QComboBox()
        for spec in ME.MODEL_FORMATS:
            self.fmt.addItem(spec["title"], spec["key"])
        self.fmt.currentIndexChanged.connect(self._fmt_changed)
        f.addRow("Формат", self.fmt)
        lay.addLayout(f)

        self.card = QTextBrowser()
        self.card.setMaximumHeight(130)
        lay.addWidget(self.card)

        opts = QHBoxLayout()
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        self.half = QCheckBox("FP16 (half)")
        self.int8 = QCheckBox("INT8")
        self.dynamic = QCheckBox("Динамический размер")
        self.simplify = QCheckBox("Упростить граф")
        self.simplify.setChecked(True)
        opts.addWidget(QLabel("Размер изображения:"))
        opts.addWidget(self.imgsz)
        for w in (self.half, self.int8, self.dynamic, self.simplify):
            opts.addWidget(w)
        opts.addStretch(1)
        lay.addLayout(opts)
        add_tips(self, {
            self.imgsz: "Размер входа модели. Должен совпадать с тем, на котором вы будете её запускать (обычно как при обучении).",
            self.half: ME.OPT_HELP["half"], self.int8: ME.OPT_HELP["int8"],
            self.dynamic: ME.OPT_HELP["dynamic"], self.simplify: ME.OPT_HELP["simplify"],
            self.weights: "Файл best.pt из папки runs/…/weights. Экспортируется именно он.",
            self.py: "Путь к python.exe, в котором установлен ultralytics.",
        })

        row = QHBoxLayout()
        self.go = QPushButton("Экспортировать")
        self.go.setProperty("primary", True)
        self.go.clicked.connect(self._run)
        self.open_btn = QPushButton("Показать результат в папке")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        self.graph_btn = QPushButton("Показать схему")
        self.graph_btn.setEnabled(False)
        self.graph_btn.clicked.connect(self._open_graph)
        self.code_btn = QPushButton("Примеры кода…")
        self.code_btn.setToolTip("<p>Готовые скрипты для запуска модели: ultralytics, OpenCV (ONNX), камера.</p>")
        self.code_btn.clicked.connect(self._code)
        self.calib_btn = QPushButton("Калибровочные картинки…")
        self.calib_btn.setToolTip("<p>Копирует ~100 случайных картинок из проекта (уменьшенных до размера входа) в выбранную папку. "
                                  "Они нужны для квантования модели в 8 бит: чем разнообразнее и ближе к реальной съёмке, тем лучше.</p>")
        self.calib_btn.clicked.connect(self._calibration)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        row.addWidget(self.go)
        row.addWidget(self.calib_btn)
        row.addWidget(self.open_btn)
        row.addWidget(self.graph_btn)
        row.addWidget(self.code_btn)
        row.addWidget(self.status, 1)
        lay.addLayout(row)

        self.result = QTextBrowser()
        self.result.setMaximumHeight(110)
        lay.addWidget(self.result)
        lay.addWidget(QLabel("Журнал"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        lay.addWidget(self.log, 1)
        self._fmt_changed()

    # ---- helpers
    def _guess_weights(self) -> str:
        base = Path(self.project.export.get("out_dir", "") or ".")
        if base.exists():
            found = sorted(base.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if found:
                return str(found[0])
        return ""

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Веса модели", self.weights.text(), "Веса YOLO (*.pt)")
        if p:
            self.weights.setText(p)

    def _fmt_changed(self):
        spec = ME.BY_KEY[self.fmt.currentData()]
        self.card.setHtml(f"<b>{spec['title']}</b><br>{spec['where']}<br><span style='color:#8b92a1'>{spec['tip']}</span>")
        if spec.get('imgsz'):
            self.imgsz.setValue(spec['imgsz'])
        if hasattr(self, 'calib_btn'):
            self.calib_btn.setVisible(spec['key'] == 'esp32')
        self.half.setEnabled("half" in spec["opts"])
        self.int8.setEnabled("int8" in spec["opts"])
        self.dynamic.setEnabled("dynamic" in spec["opts"])
        self.simplify.setEnabled("simplify" in spec["opts"])
        for w in (self.half, self.int8, self.dynamic):
            if not w.isEnabled():
                w.setChecked(False)

    @property
    def running(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    # ---- запуск
    def _run(self):
        w = Path(self.weights.text().strip())
        if not w.exists() or w.suffix.lower() != ".pt":
            QMessageBox.warning(self, "Экспорт модели", "Укажите существующий файл весов .pt (best.pt).")
            return
        py = self.py.text().strip()
        if not py:
            QMessageBox.warning(self, "Экспорт модели", "Укажите путь к Python.")
            return
        key = self.fmt.currentData()
        cmd = ME.build_command(py, str(w), key, self.imgsz.value(), self.half.isChecked(), self.int8.isChecked(),
                               self.dynamic.isChecked(), self.simplify.isChecked())
        self._log = ""
        self.log.clear()
        self.result.setHtml("")
        self.result_path = None
        self.open_btn.setEnabled(False)
        self.proc = QProcess(self)
        env = paths.child_qenv()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_out)
        self.proc.finished.connect(lambda code, _s: self._on_done(code, key))
        self.proc.setWorkingDirectory(str(w.parent))
        with paths.child_launch():           # без каталога DLL приложения (иначе Python падает с 0xC0000005)
            self.proc.start(cmd[0], cmd[1:])
            self.proc.waitForStarted(8000)
        self.go.setEnabled(False)
        self.status.setText("идёт экспорт… (первый раз может докачать нужные пакеты)")

    def _on_out(self):
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        self._log += text
        for ln in re.split(r"[\r\n]+", text):
            if ln.strip():
                self.log.appendPlainText(ln.rstrip())

    def _on_done(self, code, key):
        self.go.setEnabled(True)
        res = ME.parse_result(self._log)
        if code == 0 and res and Path(res).exists():
            self.result_path = Path(res)
            p = self.result_path
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size)
            self.status.setText("готово")
            self.open_btn.setEnabled(True)
            self.graph_btn.setEnabled(p.suffix.lower() == ".onnx")
            self.result.setHtml(cards_html([
                ("ok", f"Модель экспортирована ({size / 1048576:.1f} МБ)", f"<code>{p}</code>"),
                ("info", "Что дальше", ME.BY_KEY[key]["next"])]))
        else:
            self.status.setText("не удалось")
            self.result.setHtml(cards_html([("warn", "Экспорт не удался", ME.explain_failure(key, self._log))]))

    def _calibration(self):
        if not self.project.images:
            QMessageBox.information(self, "Калибровка", "В проекте нет изображений.")
            return
        d = QFileDialog.getExistingDirectory(self, "Папка для калибровочных картинок")
        if not d:
            return
        n = ME.export_calibration([e.path for e in self.project.images], Path(d) / "calibration",
                                  self.imgsz.value(), 100)
        self.status.setText(f"Готово: {n} картинок в {Path(d) / 'calibration'}")

    def _code(self):
        from .help_dialogs import CodeSnippetsDialog
        from .live_test import class_names_for
        w = self.weights.text().strip()
        names = class_names_for(w, self.project) if w else self.project.class_names()
        CodeSnippetsDialog("best.onnx", names, self.imgsz.value(), self).exec()

    def _open_graph(self):
        from .model_viewer import show_in_netron
        if self.result_path:
            show_in_netron(self, str(self.result_path))

    def _open_folder(self):
        if self.result_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_path.parent)))

    def reject(self):
        if self.running:
            self.proc.kill()
            self.proc.waitForFinished(2000)
        super().reject()
