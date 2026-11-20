"""Мелочи качества: контраст тем, значки стрелок и галочки, Ctrl+Y, перекраска текста, экспорт «всё или ничего»."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["MLIDE_SETTINGS_FILE"] = os.path.join(tempfile.gettempdir(), "mlide_test_polish.ini")
os.environ["ML_IDE_NO_TOUR"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from mlide.formats import Cancelled, ExportOptions, export_dataset  # noqa: E402
from mlide.model import Annotation, Project  # noqa: E402
from mlide.ui import style, winflags  # noqa: E402
from mlide.ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
winflags.install()


def lum(hexc):
    c = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_contrast():
    for name, th in style.THEMES.items():
        assert contrast("#ffffff", th["ACCENT"]) >= 4.5, (name, "белый текст на акценте")
        assert contrast(th["DISABLED"], th["PANEL"]) >= 3.0, (name, "отключённый текст")
    for dark_hex, light_hex in style.LIGHT_FIX.items():
        assert contrast(light_hex, "#ffffff") >= 4.5, (dark_hex, light_hex, contrast(light_hex, "#ffffff"))
        assert contrast(light_hex, style.THEMES["light"]["PANEL"]) >= 4.0, (light_hex, "на панели светлой темы")
    print("ok contrast")


def test_recolor_and_images():
    style.apply_theme(app, "light")
    lab = QLabel("<span style='color:#ff6b6b'>ошибка</span>")
    assert "#c62828" in lab.text() and "#ff6b6b" not in lab.text()
    lab.setText("<b style='color:#f0a030'>x</b>")
    assert "#8a5a00" in lab.text()
    for img in ("arrow_up", "arrow_down", "check"):
        assert Path(style._make_images()[img]).exists(), img
    style.apply_theme(app, "dark")
    assert style.recolor("color:#ff6b6b") == "color:#ff6b6b", "в тёмной теме цвета не меняются"
    assert "url(" in app.styleSheet() and "arrow_up" in app.styleSheet()
    print("ok recolor/images")


def test_redo_shortcut():
    w = MainWindow()
    keys = [s.toString() for s in w.a_redo.shortcuts()]
    assert "Ctrl+Y" in keys and "Ctrl+Shift+Z" in keys, keys
    print("ok Ctrl+Y")


def test_atomic_export():
    d = Path(tempfile.mkdtemp())
    pr = Project("t")
    pr.add_class("a")
    for i in range(6):
        p = d / "src" / f"i{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), np.full((40, 60, 3), 90, np.uint8))
    pr.add_images(sorted((d / "src").glob("*.jpg")))
    for e in pr.images:
        e.anns.append(Annotation(0, "box", [[5, 5], [30, 30]]))
    out = d / "out"
    o = ExportOptions(fmt="ultra_det", out_dir=str(out), train=0.8, val=0.2, epochs=1, write_train_script=False)
    export_dataset(pr, o)
    before = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert before and (out / "data.yaml").exists()
    # отмена посреди экспорта: прежний набор цел, временной папки не остаётся
    calls = {"n": 0}

    def cancel_midway(i, n, m):
        calls["n"] += 1
        return calls["n"] < 3

    try:
        export_dataset(pr, o, cancel_midway)
        raise AssertionError("ожидалась отмена")
    except Cancelled:
        pass
    after = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert after == before, "после отмены прежний экспорт должен остаться без изменений"
    assert not list(d.glob(".*mlide_tmp*")), "временная папка должна удаляться"
    export_dataset(pr, o)                                                  # повторный успешный экспорт заменяет прежний
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()) == before
    print("ok atomic export")


if __name__ == "__main__":
    test_contrast()
    test_recolor_and_images()
    test_redo_shortcut()
    test_atomic_export()
    print("ALL OK")
