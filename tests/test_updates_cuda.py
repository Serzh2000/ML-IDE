"""Проверка обновлений и помощника по CUDA (логика без сети и без установки PyTorch). Запуск: python tests/test_updates_cuda.py"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MLIDE_SETTINGS_FILE", os.path.join(__import__("tempfile").gettempdir(), "mlide_test_settings.ini"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlide import updates
from mlide.ui import help_dialogs as H


def test_versions():
    assert updates.parse_version("v0.4.0") == (0, 4, 0)
    assert updates.is_newer("0.4.0", "v0.4.1") and updates.is_newer("0.9.9", "0.10.0")
    assert not updates.is_newer("0.4.0", "v0.4.0") and not updates.is_newer("1.0", "0.9.9")
    assert updates.is_newer("0.4", "0.4.0.1")


def test_check_with_fake_network():
    fake = {"tag": "v9.9.9", "name": "x", "url": "https://example.invalid/r", "body": "notes", "assets": []}
    r = updates.check("0.4.0", fetch=lambda: fake)
    assert r["status"] == "new" and r["latest"]["tag"] == "v9.9.9"
    assert updates.check("9.9.9", fetch=lambda: fake)["status"] == "current"

    def boom():
        raise OSError("offline")
    assert updates.check("0.4.0", fetch=boom)["status"] == "error"


def test_cuda_plan():
    exists = {130, 128, 126, 124, 121, 118}
    probe = lambda tag: tag in exists
    assert H.parse_cuda("12.8") == 128 and H.parse_cuda("13.0") == 130 and H.parse_cuda("?") == 0
    p = H.torch_cuda_plan("12.9", probe)                      # 129 нет -> берём 128
    assert p["ok"] and p["tag"] == "cu128" and p["install"][-1].endswith("/cu128")
    assert H.torch_cuda_plan("12.5", probe)["tag"] == "cu124"
    assert H.torch_cuda_plan("11.7", probe)["ok"] is False
    assert H.torch_cuda_plan("", probe)["ok"] is False
    assert "uninstall" in H.torch_cuda_plan("12.4", probe)["uninstall"]


def test_install_chain_runs_commands():
    """Цепочка шагов установки выполняется по порядку (вместо pip - безобидные команды Python)."""
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    d = H.GpuHelpDialog()
    py = sys.executable
    d._steps = [(py, ["-c", "print('step1')"]), (py, ["-c", "print('step2')"]),
                (py, ["-c", "print('torch X | CUDA доступна: True | fake')"])]
    d.install_btn.setEnabled(False)
    d._next_step()
    t = time.time()
    while time.time() - t < 30 and not d.install_btn.isEnabled():
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec()
    log = d.log.toPlainText()
    assert "step1" in log and "step2" in log and "Готово" in log, log
    # неудача шага останавливает цепочку
    d2 = H.GpuHelpDialog()
    d2._steps = [(py, ["-c", "import sys; sys.exit(3)"]), (py, ["-c", "print('must not run')"])]
    d2.install_btn.setEnabled(False)
    d2._next_step()
    t = time.time()
    while time.time() - t < 30 and not d2.install_btn.isEnabled():
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec()
    assert "must not run" not in d2.log.toPlainText() and "кодом 3" in d2.log.toPlainText(), d2.log.toPlainText()


def test_update_dialog_renders():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    d = H.UpdateDialog.__new__(H.UpdateDialog)               # без сети: проверяем только отрисовку результата
    H.QDialog.__init__(d)
    d.view = H.QTextBrowser()
    d.open_btn = H.QPushButton()
    d.url = ""
    H.UpdateDialog._done(d, {"status": "new", "latest": {"tag": "v1.0.0", "url": "https://x", "body": "<b>hi</b>\nnotes",
                                                           "assets": [{"name": "a.exe", "url": "u", "size": 5 << 20}]}})
    assert "v1.0.0" in d.view.toPlainText() and d.open_btn.isEnabled()
    H.UpdateDialog._done(d, {"status": "error", "error": "OSError: offline"})
    assert "offline" in d.view.toPlainText()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL OK")
