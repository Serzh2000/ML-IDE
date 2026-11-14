"""Делает скриншоты для README (нужен рабочий стол, не offscreen: иначе не будет шрифтов).

    python tools/make_screenshots.py --project <wildlife.mlide.json> --seg-project <cracks.mlide.json> ^
           --train-folder <папка экспорта с runs/> --out docs/screenshots
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["ML_IDE_NO_TOUR"] = "1"

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog

from mlide import paths
from mlide.ui.dialogs import AutoLabelDialog, ExportDialog, PresetDialog
from mlide.ui.examples_dialog import ExamplesDialog
from mlide.ui.learn import AdvisorDialog, HelpDialog, TrainDialog
from mlide.ui.main_window import SHORTCUTS_HTML, MainWindow, install_translations
from mlide.ui.results import ResultsDialog
from mlide.ui.style import apply_dark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--seg-project")
    ap.add_argument("--train-folder", required=True)
    ap.add_argument("--test-image")
    ap.add_argument("--out", default="docs/screenshots")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # в скриншотах не должно быть личных путей: показываем нейтральные
    paths.examples_dir = lambda: Path("C:/Users/user/ML IDE/examples")
    app = QApplication([])
    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(str(paths.resource("assets/icon.ico"))))
    install_translations(app)
    apply_dark(app)

    def settle(ms=350):
        # QEventLoop, а не app.quit(): quit() закрывает все окна
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def shot(w, name):
        settle()
        w.grab().save(str(out / name))
        print("saved", name)

    w = MainWindow()
    w.resize(1500, 920)
    w.show()
    shot(w, "welcome.png")

    def pick(project_window):
        pr = project_window.project
        best = max((i for i, e in enumerate(pr.images) if 2 <= len(e.anns) <= 6), key=lambda i: -abs(len(pr.images[i].anns) - 3),
                   default=0)
        project_window.load_image(best)

    w._open_path(a.project)
    pick(w)
    w.set_tool("select")
    w.canvas.select(w.entry.anns[0].id)
    shot(w, "main.png")

    if a.seg_project:
        w._open_path(a.seg_project)
        pick(w)
        w.canvas.select(w.entry.anns[0].id)
        shot(w, "segmentation.png")
        w._open_path(a.project)
        pick(w)

    w.project.export["out_dir"] = a.train_folder
    for name, dlg in (("advisor.png", AdvisorDialog(w.project, w)),
                      ("export.png", ExportDialog(w.project, w)),
                      ("presets.png", PresetDialog(w.project, w.entry, w, current="Сильная")),
                      ("autolabel.png", AutoLabelDialog(w.project, w.entry, w)),
                      ("help.png", HelpDialog(SHORTCUTS_HTML, w, "Как читать")),
                      ("examples.png", ExamplesDialog(w))):
        dlg.show()
        shot(dlg, name)
        dlg.close()

    tr = TrainDialog(w.project, w)
    tr.py.setText("python")
    tr.show()
    shot(tr, "training.png")
    tr.close()

    rd = ResultsDialog(w.project, w)
    rd.show()
    shot(rd, "results.png")
    if a.test_image:
        QFileDialog.getOpenFileName = staticmethod(lambda *x, **k: (a.test_image, ""))
        rd._test_model()
        rd.tabs.setCurrentIndex(1)
        shot(rd, "results_check.png")
    rd.close()

    w.start_tour()
    for step in (3, 8):
        w._tour.i = step
        w._tour.show_step()
        shot(w, f"tour_{step + 1}.png")
    w._tour.finish()
    print("done")


if __name__ == "__main__":
    main()
