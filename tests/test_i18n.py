"""Проверка английского интерфейса: в окнах не должно оставаться русского текста.

Запуск:  MLIDE_LANG=en python tests/test_i18n.py     (на Windows: set MLIDE_LANG=en)
"""
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ["MLIDE_LANG"] = "en"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from mlide import i18n, imgio
from mlide.model import Annotation, Project

i18n.set_language("en")
assert i18n.lang() == "en" and i18n.load_table(), "словарь перевода не найден (assets/i18n_en.json)"

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QListWidget, QMenu, QTabWidget, QTableWidget, QTextEdit,
                               QTreeWidget, QWidget, QAbstractButton, QGroupBox, QLineEdit)

from mlide.ui.compare_dialog import CompareDialog
from mlide.ui.dedupe_dialog import DedupeDialog
from mlide.ui.dialogs import AutoLabelDialog, ExportDialog, PresetDialog, StatsDialog
from mlide.ui.examples_dialog import ExamplesDialog
from mlide.ui.help_dialogs import CodeSnippetsDialog, GpuHelpDialog
from mlide.ui.learn import AdvisorDialog, HelpDialog, TrainDialog
from mlide.ui.live_test import LiveTestDialog
from mlide.ui.main_window import SHORTCUTS_HTML, MainWindow
from mlide.ui.model_export import ModelExportDialog
from mlide.ui.model_viewer import ModelViewerDialog
from mlide.ui.results import ResultsDialog
from mlide.ui.review_dialog import ReviewDialog
from mlide.ui.video_dialog import VideoDialog

CYR = re.compile(r"[А-Яа-яЁё]")


def texts(root):
    """Все видимые тексты окна: подписи, кнопки, подсказки, меню, таблицы, списки, вкладки, HTML."""
    out = []
    widgets = [root] + root.findChildren(QWidget)
    for w in widgets:
        out.append(("title", w.windowTitle()))
        out.append(("tip", w.toolTip()))
        if isinstance(w, (QLabel, QAbstractButton)):
            out.append((type(w).__name__, w.text()))
        if isinstance(w, QGroupBox):
            out.append(("group", w.title()))
        if isinstance(w, QLineEdit):
            out.append(("placeholder", w.placeholderText()))
        if isinstance(w, QTextEdit):
            out.append(("html", w.toPlainText()))
        if isinstance(w, QComboBox):
            out += [("combo", w.itemText(i)) for i in range(w.count())]
        if isinstance(w, QListWidget):
            out += [("list", w.item(i).text()) for i in range(w.count())]
        if isinstance(w, QTabWidget):
            out += [("tab", w.tabText(i)) for i in range(w.count())]
        if isinstance(w, QTableWidget):
            out += [("header", (w.horizontalHeaderItem(c).text() if w.horizontalHeaderItem(c) else "")) for c in range(w.columnCount())]
        if isinstance(w, QTreeWidget):
            h = w.headerItem()
            out += [("header", h.text(c)) for c in range(w.columnCount())]
    for a in root.findChildren(QAction):
        out += [("action", a.text()), ("action-tip", a.toolTip())]
    for m in root.findChildren(QMenu):
        out.append(("menu", m.title()))
    return [(k, t) for k, t in out if t and CYR.search(t)]


def main():
    app = QApplication([])
    from mlide.ui.style import apply_dark
    apply_dark(app)
    tmp = Path(tempfile.mkdtemp(prefix="mlide_i18n_"))
    pr = Project("t")
    pr.add_class("thing")
    for i in range(6):
        im = np.full((120, 160, 3), 70 + i * 10, np.uint8)
        imgio.imwrite(tmp / f"img_{i:03d}.jpg", im)
    pr.add_images(sorted(tmp.glob("*.jpg")))
    for e in pr.images[:5]:
        e.anns.append(Annotation(0, "box", [[10, 10], [80, 80]]))
    pr.save(tmp / "p.mlide.json")

    w = MainWindow()
    w.show()
    w._open_path(str(tmp / "p.mlide.json"))
    app.processEvents()
    wins = {
        "main window": w, "advisor": AdvisorDialog(pr, w), "export": ExportDialog(pr, w), "autolabel": AutoLabelDialog(pr, pr.images[0], w),
        "presets": PresetDialog(pr, pr.images[0], w), "stats": StatsDialog(pr, w), "examples": ExamplesDialog(w),
        "help": HelpDialog(SHORTCUTS_HTML, w), "train": TrainDialog(pr, w), "results": ResultsDialog(pr, w),
        "model export": ModelExportDialog(pr, w), "model viewer": ModelViewerDialog(w), "video": VideoDialog(w),
        "duplicates": DedupeDialog(pr, w), "review": ReviewDialog(pr, w), "compare": CompareDialog(pr, w),
        "live test": LiveTestDialog(pr, w), "gpu": GpuHelpDialog(w), "code": CodeSnippetsDialog("best.onnx", ["thing"], 640, w),
    }
    # логика не должна зависеть от языка: пресеты ищутся по скрытому ключу, а не по отображаемому тексту
    assert wins["export"].preset.currentData() == "Средняя (по умолчанию YOLO)", wins["export"].preset.currentData()
    assert wins["presets"].current_name in ("Средняя (по умолчанию YOLO)",), wins["presets"].current_name
    wins["export"].preset.setCurrentIndex(wins["export"].preset.findData("Сильная"))
    assert wins["export"].preset.currentData() == "Сильная"
    w.start_tour()
    for i in range(len(w._tour.steps)):
        w._tour.i = i
        w._tour.show_step()
        wins[f"tour step {i + 1}"] = w._tour
    # справочник целиком: все темы
    hd = wins["help"]
    for r in range(hd.lst.count()):
        it = hd.lst.item(r)
        if it.data(256):
            hd.lst.setCurrentRow(r)
            wins[f"help topic {r}"] = hd.view

    bad = []
    for name, win in wins.items():
        if hasattr(win, "toHtml") and not isinstance(win, QWidget):
            continue
        items = texts(win) if name.startswith(("main", "advisor", "export", "auto", "presets", "stats", "examples", "train", "results",
                                              "model", "video", "duplicates", "review", "compare", "live", "gpu", "code", "tour")) else []
        if name.startswith("help topic") and hasattr(win, "toPlainText") and CYR.search(win.toPlainText()):
            items = [("html", win.toPlainText())]
        for kind, t in items:
            bad.append((name, kind, t[:90].replace("\n", " ")))
    # меню главного окна
    for m in w.menuBar().findChildren(QMenu):
        for a in m.actions():
            if CYR.search(a.text()):
                bad.append(("main menu", "action", a.text()))
    print(f"окон проверено: {len(wins)}; осталось русских фрагментов: {len(bad)}")
    for b in bad[:60]:
        print("  ", b)
    limit = int(os.environ.get("I18N_MAX_LEFT", "0"))
    assert len(bad) <= limit, f"в английском интерфейсе осталось {len(bad)} русских текстов"
    print("I18N OK")


if __name__ == "__main__":
    main()
