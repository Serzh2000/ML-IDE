"""Окна: одно на тип с переводом фокуса при повторном открытии, закрытие при смене проекта, возможность менять размер."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["MLIDE_SETTINGS_FILE"] = os.path.join(tempfile.gettempdir(), "mlide_test_windows.ini")
os.environ["ML_IDE_NO_TOUR"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from mlide.ui import winflags  # noqa: E402
from mlide.ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
winflags.install()


def test_unique_windows():
    w = MainWindow()
    w.show()
    w.compare_runs()
    d1 = w._unique["compare"]
    assert d1.isVisible() and not d1.isModal()
    w.compare_runs()
    assert w._unique["compare"] is d1, "повторное открытие должно возвращать то же окно"
    w.live_test()
    w.gpu_help()
    assert {"compare", "livetest", "gpu"} <= set(w._unique)
    w.stats()                                    # окно со снимком данных открывается заново (актуальные цифры)
    s1 = w._unique["stats"]
    w.stats()
    assert w._unique["stats"] is not s1
    w._close_unique()
    assert not w._unique
    w.compare_runs()
    d2 = w._unique["compare"]
    w.new_project()                              # смена проекта закрывает окна старого проекта
    assert not d2.isVisible() or not w._unique
    print("ok unique windows")


def test_resizable():
    w = MainWindow()
    from mlide.ui.dialogs import StatsDialog
    d = StatsDialog(w.project, w)
    assert d.isSizeGripEnabled()
    assert d.windowFlags() & Qt.WindowMaximizeButtonHint
    assert isinstance(d, QDialog)
    print("ok resizable")


if __name__ == "__main__":
    test_unique_windows()
    test_resizable()
    print("ALL OK")
