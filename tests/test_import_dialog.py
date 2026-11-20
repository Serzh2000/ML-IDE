"""Мастер импорта разметки: определение формата, правка столбцов таблицы, сопоставление классов, запись в проект (Qt, offscreen)."""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["MLIDE_SETTINGS_FILE"] = os.path.join(tempfile.gettempdir(), "mlide_test_import_dialog.ini")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from mlide.model import Project  # noqa: E402
from mlide.ui.import_dialog import SKIP_CLASS, ImportDialog  # noqa: E402

app = QApplication([])
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)


def wait_until(cond, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        l = QEventLoop()
        QTimer.singleShot(50, l.quit)
        l.exec()
        if cond():
            return True
    return False


def img(path, w=200, h=100):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 100, np.uint8))


def test_yolo_via_dialog():
    d = Path(tempfile.mkdtemp())
    img(d / "images/train/a.jpg")
    img(d / "images/train/b.jpg")
    (d / "labels/train").mkdir(parents=True)
    (d / "labels/train/a.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n")
    (d / "labels/train/b.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    (d / "data.yaml").write_text("names:\n  0: cat\n  1: dog\n")
    pr = Project("t")
    dlg = ImportDialog(pr, paths=[str(d)])
    assert wait_until(lambda: dlg.pv is not None), "предпросмотр не построился"
    assert dlg.current_det().parser == "yolo"
    assert dlg.pv.records == 2 and dlg.pv.matched == 2 and dlg.pv.objects == 3, dlg.pv
    assert dlg.cls_table.rowCount() == 2 and dlg.go.isEnabled()
    # пропускаем класс dog, переименовываем cat
    for r in range(dlg.cls_table.rowCount()):
        src = dlg.cls_table.item(r, 0).data(0x100)
        cb = dlg.cls_table.cellWidget(r, 2)
        if src == "dog":
            cb.setCurrentIndex(cb.findData("skip"))
        else:
            cb.setEditText("кошка")
    dlg._run()
    assert wait_until(lambda: dlg.report is not None), "запись не завершилась"
    assert dlg.report.images_added == 2 and dlg.report.objects_added == 1, dlg.report
    assert pr.class_names() == ["кошка"], pr.class_names()
    print("ok yolo dialog")


def test_table_mapping_edit():
    d = Path(tempfile.mkdtemp())
    img(d / "pics/a.jpg")
    (d / "weird.csv").write_text("pic,thing,aa,bb,cc,dd\na.jpg,box1,10,20,30,40\n")
    pr = Project("t")
    dlg = ImportDialog(pr, paths=[str(d / "weird.csv")], images_dir=d / "pics")
    assert wait_until(lambda: dlg.pv is not None)
    assert dlg.current_det().parser == "table"
    # автоопределение не знает aa..dd: ставим столбцы вручную
    dlg.pv = None
    dlg.c_geo.setCurrentIndex(dlg.c_geo.findData("xywh"))
    for cb, name in zip(dlg.c_cols, ("aa", "bb", "cc", "dd")):
        cb.setCurrentIndex(cb.findData(name))
    dlg.c_class.setCurrentIndex(dlg.c_class.findData("thing"))
    assert wait_until(lambda: dlg.pv is not None and dlg.pv.objects == 1), dlg.pv
    dlg._run()
    assert wait_until(lambda: dlg.report is not None)
    assert pr.images[0].anns[0].pts == [[10, 20], [40, 60]], pr.images[0].anns[0].pts
    print("ok table mapping dialog")


def test_nothing_found():
    d = tempfile.mkdtemp()
    dlg = ImportDialog(Project("t"), paths=[d])
    assert wait_until(lambda: dlg.status.text() == "" and not dlg._busy)
    assert dlg.fmt.count() == 0 and not dlg.go.isEnabled()
    print("ok nothing found")


if __name__ == "__main__":
    test_yolo_via_dialog()
    test_table_mapping_edit()
    test_nothing_found()
    print("ALL OK")
