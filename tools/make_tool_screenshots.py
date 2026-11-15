"""Скриншоты новых инструментов для README (нужен рабочий стол). Пример:

    python tools/make_tool_screenshots.py --project wildlife.mlide.json --model best.onnx --runs-folder <папка с runs/> ^
           --sam-models <папка с mobile_sam_*.onnx> --out docs/screenshots
"""
import argparse
import copy
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["ML_IDE_NO_TOUR"] = "1"
os.environ.setdefault("MLIDE_SETTINGS_FILE", os.path.join(__import__("tempfile").gettempdir(), "mlide_shots_settings.ini"))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from mlide import i18n, paths, sam
from mlide.model import Project
from mlide.ui.compare_dialog import CompareDialog
from mlide.ui.dedupe_dialog import DedupeDialog
from mlide.ui.help_dialogs import CodeSnippetsDialog
from mlide.ui.live_test import LiveTestDialog
from mlide.ui.main_window import MainWindow, install_translations
from mlide.ui.review_dialog import ReviewDialog
from mlide.ui.style import apply_theme, detect_theme
from mlide.ui.video_dialog import VideoDialog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--model", required=True, help="best.onnx (путь для показа в окнах; лучше нейтральный)")
    ap.add_argument("--runs-folder", required=True)
    ap.add_argument("--sam-models", help="папка с mobile_sam_image_encoder.onnx и sam_mask_decoder_single.onnx")
    ap.add_argument("--out", default="docs/screenshots")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    i18n.set_language(i18n.detect_language())
    if a.sam_models:
        sam.models_dir = lambda: Path(a.sam_models)
    app = QApplication([])
    if i18n.lang() == "ru":
        install_translations(app)
    apply_theme(app, detect_theme())

    def settle(ms=500):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def shot(w, name):
        settle()
        w.grab().save(str(out / name))
        print("saved", name)

    pr = Project.load(a.project)
    pr.export["out_dir"] = a.runs_folder
    pr.export["auto_model"] = a.model

    # --- SAM: клик -> предпросмотр контура
    w = MainWindow()
    w.resize(1500, 920)
    w.show()
    w._open_path(a.project)
    best = max((i for i, e in enumerate(w.project.images) if len(e.anns) <= 2), key=lambda i: w.project.images[i].w, default=0)
    w.load_image(best)
    w.set_tool("sam")
    e = w.entry
    if a.sam_models:
        w.canvas._sam_pts.append((e.w * 0.5, e.h * 0.5, 1))
        w.canvas.samPoints.emit(list(w.canvas._sam_pts))
        shot(w, "sam.png")
    w.close()

    # --- дубли: берём 12 кадров и добавляем «почти копии» (пересжатие JPEG)
    tmp = Path(tempfile.mkdtemp(prefix="mlide_dup_"))
    import cv2
    from mlide import imgio
    src = [e.path for e in pr.images[:12]]
    dup = Project("dups")
    for i, p in enumerate(src):
        shutil.copy(p, tmp / f"a_{i:02d}.jpg")
        if i % 3 == 0:
            im = imgio.imread(p)
            imgio.imwrite(tmp / f"copy_of_{i:02d}.jpg", cv2.resize(im, None, fx=0.8, fy=0.8), quality=60)
    dup.add_images(sorted(tmp.glob("*.jpg")))
    d = DedupeDialog(dup, None)
    d.resize(760, 520)
    d.show()
    d._run()
    for _ in range(60):
        settle(200)
        if d.groups:
            break
    shot(d, "duplicates.png")
    d.close()

    # --- проверка разметки моделью
    sub = Project("sub")
    sub.classes = copy.deepcopy(pr.classes)
    sub.images = copy.deepcopy(pr.images[:60])
    # намеренно портим разметку, чтобы показать поиск ошибок: не тот класс, потерянная рамка, сдвинутая рамка
    n = len(sub.classes)
    for e in sub.images[:5]:
        for an in e.anns:
            an.cls = (an.cls + 1) % n
    for e in sub.images[5:10]:
        if len(e.anns) > 1:
            e.anns.sort(key=lambda an: -(an.bbox()[2] - an.bbox()[0]) * (an.bbox()[3] - an.bbox()[1]))
            e.anns.pop(0)
    for e in sub.images[10:14]:
        for an in e.anns:
            an.pts = [[x + 0.35 * e.w, y + 0.2 * e.h] for x, y in an.pts]
    r = ReviewDialog(sub, None)
    r.path.setText(a.model)
    r.mode.setCurrentIndex(0)
    r.show()
    r._run()
    for _ in range(100):
        settle(200)
        if r.go.isEnabled() and r.table.rowCount():
            break
    shot(r, "review.png")
    r.close()

    # --- сравнение запусков (реальный + запуск на малых данных, если есть)
    c = CompareDialog(pr, None)
    c.show()
    c.table.selectAll()
    shot(c, "compare.png")
    c.close()

    # --- тест на видео / замер скорости
    lt = LiveTestDialog(pr, None)
    lt.model.setText(a.model)
    lt.src.setCurrentText("0")
    lt.show()
    lt._bench()
    for _ in range(100):
        settle(200)
        if lt.benchbtn.isEnabled() and "кадр" in lt.stats.text() or "FPS" in lt.stats.text() or "fps" in lt.stats.text():
            break
    shot(lt, "live_test.png")
    lt.close()

    v = VideoDialog(None)
    v.show()
    shot(v, "video.png")
    v.close()

    cs = CodeSnippetsDialog("best.onnx", pr.class_names(), 640, None)
    cs.tabs.setCurrentIndex(1)
    cs.show()
    shot(cs, "code.png")
    cs.close()
    print("done")


if __name__ == "__main__":
    main()
