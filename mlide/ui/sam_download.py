"""Диалог первого запуска «Разметки по клику»: согласие и загрузка модели MobileSAM (~43 МБ)."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from .. import sam


class _Worker(QThread):
    progress = Signal(str, int, int)
    done = Signal()
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            sam.download_models(lambda n, d, t: (self.progress.emit(n, d, t), not self._cancel)[1])
            self.done.emit()
        except Exception as e:  # noqa
            self.failed.emit(str(e))


def ensure_sam(parent) -> bool:
    """True, если модель готова к работе (при необходимости спрашивает разрешение и скачивает)."""
    if not sam.runtime_available():
        QMessageBox.warning(parent, "Разметка по клику", "Не найден пакет onnxruntime.\nУстановите: pip install onnxruntime")
        return False
    if sam.models_ready():
        return True
    r = QMessageBox.question(
        parent, "Разметка по клику (SAM)",
        "Для этой функции нужна нейросеть MobileSAM (≈43 МБ). Она скачивается один раз с Hugging Face "
        f"(Acly/MobileSAM) в папку\n{sam.models_dir()}\nи дальше работает без интернета.\n\nСкачать сейчас?")
    if r != QMessageBox.Yes:
        return False
    dlg = QProgressDialog("Загрузка модели MobileSAM…", "Отмена", 0, 0, parent)
    dlg.setWindowTitle("Загрузка")
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    w = _Worker()
    result = {"ok": False, "err": ""}

    def on_progress(name, done, total):
        if total:
            dlg.setMaximum(total)
            dlg.setValue(done)
        dlg.setLabelText(f"Загрузка {name}…")

    w.progress.connect(on_progress)
    w.done.connect(lambda: (result.update(ok=True), dlg.close()))
    w.failed.connect(lambda m: (result.update(err=m), dlg.close()))
    dlg.canceled.connect(w.cancel)
    w.start()
    dlg.exec()
    w.wait()
    if not result["ok"]:
        if result["err"] and "Отменено" not in result["err"]:
            QMessageBox.warning(parent, "Загрузка модели", result["err"])
        return False
    return True
