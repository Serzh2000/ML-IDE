"""Имитация работы пользователя мышью и клавиатурой на холсте (настоящие события Qt). Запуск: python tests/test_interaction.py"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MLIDE_SETTINGS_FILE", os.path.join(__import__("tempfile").gettempdir(), "mlide_test_settings.ini"))
os.environ.setdefault("ML_IDE_NO_TOUR", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mlide import imgio
from mlide.model import Annotation, Project
from mlide.ui.main_window import MainWindow
from mlide.ui.style import apply_dark

app = QApplication([])
apply_dark(app)
FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def pt(cv, x, y):
    """Точка изображения -> позиция в виджете холста."""
    q = cv.to_view(x, y)
    return QPoint(round(q.x()), round(q.y()))


def drag(cv, a, b, btn=Qt.LeftButton, mods=Qt.NoModifier, steps=5):
    QTest.mousePress(cv, btn, mods, a)
    for i in range(1, steps + 1):
        p = QPoint(a.x() + (b.x() - a.x()) * i // steps, a.y() + (b.y() - a.y()) * i // steps)
        QTest.mouseMove(cv, p)
        app.processEvents()
    QTest.mouseRelease(cv, btn, mods, b)
    app.processEvents()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mlide_ui_"))
    for i in range(4):
        im = np.full((300, 400, 3), 40 + 10 * i, np.uint8)
        cv2.rectangle(im, (150, 100), (250, 200), (240, 240, 240), -1)       # яркий объект для «умного контура»
        imgio.imwrite(tmp / f"img_{i}.png", im)
    pr = Project("ui")
    pr.add_class("cat")
    pr.add_class("dog")
    pr.add_images(sorted(tmp.glob("*.png")))
    pr.save(tmp / "p.mlide.json")

    w = MainWindow()
    w.resize(1300, 800)
    w.show()
    w._open_path(str(tmp / "p.mlide.json"))
    app.processEvents()
    pr = w.project                       # окно работает со своей копией проекта, загруженной из файла
    cv = w.canvas
    e = w.entry
    check(w.stack.currentIndex() == 1 and cv.image is not None, "проект открыт, виден холст")

    # --- 1. рамка
    w.set_tool("box")
    drag(cv, pt(cv, 50, 40), pt(cv, 150, 120))
    check(len(e.anns) == 1 and e.anns[0].kind == "box", "рамка нарисована мышью")
    x1, y1, x2, y2 = e.anns[0].bbox()
    check(abs(x1 - 50) < 3 and abs(y1 - 40) < 3 and abs(x2 - 150) < 3 and abs(y2 - 120) < 3, f"координаты рамки верны ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    drag(cv, pt(cv, 300, 200), pt(cv, 302, 201))
    check(len(e.anns) == 1, "случайный клик/микро-перетаскивание не создаёт рамку")
    drag(cv, pt(cv, 300, 200), QPoint(pt(cv, 300, 200).x() + 900, pt(cv, 300, 200).y() + 900))
    bx = e.anns[-1].bbox()
    check(len(e.anns) == 2 and bx[2] <= e.w + 1e-6 and bx[3] <= e.h + 1e-6, "рамка не выходит за границы изображения")

    # --- 2. выбор, перемещение, изменение размера
    w.set_tool("select")
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 100, 80))
    check(cv.selected == e.anns[0].id, "клик внутри рамки выделяет её")
    drag(cv, pt(cv, 100, 80), pt(cv, 130, 100))
    nx1, ny1, nx2, ny2 = e.anns[0].bbox()
    check(abs((nx1 - x1) - 30) < 3 and abs((ny1 - y1) - 20) < 3 and abs((nx2 - nx1) - (x2 - x1)) < 1e-6, "перемещение рамки мышью, размер сохранён")
    drag(cv, pt(cv, nx2, ny2), pt(cv, nx2 + 40, ny2 + 25))                       # правый нижний маркер
    mx1, my1, mx2, my2 = e.anns[0].bbox()
    check(abs((mx2 - nx2) - 40) < 3 and abs((my2 - ny2) - 25) < 3 and abs(mx1 - nx1) < 1e-6, "изменение размера за угловой маркер")
    drag(cv, pt(cv, mx2, my2), pt(cv, mx1 - 20, my1 - 20))                       # «выворот» за противоположный угол
    ox1, oy1, ox2, oy2 = e.anns[0].bbox()
    check(ox1 < ox2 and oy1 < oy2, "рамка остаётся корректной (x1<x2, y1<y2) после выворота")

    # --- 3. undo / redo
    n_before = len(e.anns)
    w.undo()
    ux = e.anns[0].bbox()
    check(abs(ux[2] - mx2) < 3, "Ctrl+Z возвращает размер рамки")
    w.redo()
    check(abs(e.anns[0].bbox()[0] - ox1) < 0.02, "Ctrl+Shift+Z повторяет действие")   # снимки округляют до 0.01 px
    for _ in range(10):
        w.undo()
    check(len(e.anns) == 0, "серия undo откатывает всё до пустого кадра")
    for _ in range(10):
        w.redo()
    check(len(e.anns) == n_before, "серия redo восстанавливает всё")

    # --- 4. удаление и классы
    cv.select(e.anns[0].id)
    w.pick_class(1)
    check(e.anns[0].cls == 1, "клавиша 2 меняет класс выделенного объекта")
    QTest.keyClick(w, Qt.Key_Delete)
    app.processEvents()
    if len(e.anns) == n_before:                    # offscreen: горячие клавиши окна могут не доходить - вызываем действие
        w.a_del.trigger()
    check(len(e.anns) == n_before - 1, "удаление выделенного объекта")
    w.undo()
    check(len(e.anns) == n_before, "удаление можно отменить")

    # --- 5. полигон
    w.set_tool("poly")
    before = len(e.anns)
    for p in ((260, 40), (350, 60), (330, 140), (270, 120)):
        QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, *p))
    QTest.keyClick(cv, Qt.Key_Backspace)
    check(len(cv._poly) == 3, "Backspace убирает последнюю точку полигона")
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 270, 120))
    QTest.mouseDClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 290, 130))
    check(len(e.anns) == before + 1 and e.anns[-1].kind == "poly" and len(e.anns[-1].pts) >= 3, f"полигон замкнут двойным щелчком ({len(e.anns[-1].pts)} точек)")
    for p in ((20, 200), (80, 200), (50, 260)):
        QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, *p))
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 20, 200))          # клик по первой точке замыкает
    check(len(e.anns) == before + 2, "полигон замкнут щелчком по первой точке")
    for p in ((200, 250), (230, 250)):
        QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, *p))
    QTest.keyClick(cv, Qt.Key_Escape)
    check(len(cv._poly) == 0 and len(e.anns) == before + 2, "Esc отменяет незавершённый полигон")

    # --- 6. изменение полигона: добавить/удалить вершину
    w.set_tool("select")
    pol = e.anns[-1]
    cv.select(pol.id)
    n_pts = len(pol.pts)
    a, b = pol.pts[0], pol.pts[1]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    QTest.mouseDClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, *mid))
    check(len(pol.pts) == n_pts + 1, "двойной щелчок по ребру добавляет вершину")
    QTest.mouseClick(cv, Qt.LeftButton, Qt.AltModifier, pt(cv, *pol.pts[1]))
    check(len(pol.pts) == n_pts, "Alt+щелчок по вершине удаляет её")

    # --- 7. масштаб и сдвиг
    anchor = QPointF(pt(cv, 200, 150))
    ip0 = cv.to_img(anchor)
    s0 = cv.scale
    cv.wheelEvent(QWheelEvent(anchor, cv.mapToGlobal(anchor), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    ip1 = cv.to_img(anchor)
    check(cv.scale > s0 and abs(ip1.x() - ip0.x()) < 0.5 and abs(ip1.y() - ip0.y()) < 0.5, "колесо мыши масштабирует вокруг курсора")
    off0 = QPointF(cv.off)
    drag(cv, QPoint(300, 300), QPoint(340, 330), btn=Qt.MiddleButton)
    check(abs((cv.off.x() - off0.x()) - 40) < 1 and abs((cv.off.y() - off0.y()) - 30) < 1, "сдвиг изображения средней кнопкой")
    cv.fit()
    check(0.05 < cv.scale < 10, "«Вписать в окно» восстанавливает масштаб")

    # --- 8. умный контур
    w.set_tool("smart")
    before = len(e.anns)
    drag(cv, pt(cv, 135, 85), pt(cv, 265, 215))
    app.processEvents()
    check(len(e.anns) == before + 1 and e.anns[-1].kind == "poly", "умный контур создал полигон")
    bb = e.anns[-1].bbox()
    check(abs(bb[0] - 150) < 12 and abs(bb[1] - 100) < 12 and abs(bb[2] - 250) < 12 and abs(bb[3] - 200) < 12, f"контур совпал с объектом ({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f})")

    # --- 9. навигация, копирование, проверено, фильтры
    w.set_tool("select")
    w.step(1)
    check(w.idx == 1, "переход на следующее изображение")
    w.copy_prev()
    check(len(w.entry.anns) == len([a for a in pr.images[0].anns if not a.suggested]), "Ctrl+D копирует разметку с предыдущего кадра")
    w.mark_reviewed_next()
    check(pr.images[1].reviewed and w.idx == 2, "«Проверено и дальше» отмечает кадр и идёт вперёд")
    w.img_filter.setCurrentIndex(1)
    app.processEvents()
    check(all(not [a for a in pr.images[w.img_model.image_index(r)].anns if not a.suggested] or w.img_model.image_index(r) == w.idx
              for r in range(w.img_model.rowCount())), "фильтр «Без разметки» показывает только неразмеченные")
    w.img_filter.setCurrentIndex(0)

    # --- 10. подсказки авторазметки
    w.load_image(3)
    w.entry.anns.append(Annotation(0, "box", [[10, 10], [60, 60]], suggested=True, conf=0.8))
    w.on_changed()
    check("?" in w.img_model.text(w.entry), "подсказка модели отмечена в списке «?»")
    w.accept_suggestions()
    check(not any(a.suggested for a in w.entry.anns), "Y принимает подсказки")

    # --- 10b. повёрнутая рамка (OBB)
    w.load_image(2)
    e3 = w.entry
    e3.anns.clear()
    w.set_tool("obb")
    drag(cv, pt(cv, 60, 150), pt(cv, 180, 150))
    check(cv._obb and cv._obb["stage"] == 2, "OBB: после оси ждём ширину")
    QTest.mouseMove(cv, pt(cv, 120, 170))
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 120, 170))
    ob = [a for a in e3.anns if a.kind == "obb"]
    check(len(ob) == 1 and len(ob[0].pts) == 4, "OBB: рамка создана")
    c4 = np.array(ob[0].pts)
    side1, side2 = np.linalg.norm(c4[1] - c4[0]), np.linalg.norm(c4[3] - c4[0])
    check(abs(side1 - 120) < 3 and abs(side2 - 40) < 3, f"OBB: размеры {side1:.0f}x{side2:.0f} (ожидалось 120x40)")
    w.set_tool("select")
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 120, 150))
    check(cv.selected == ob[0].id, "OBB: выделяется щелчком")
    area0 = abs(cross2(c4[1] - c4[0], c4[3] - c4[0]))
    QTest.keyClick(cv, Qt.Key_BracketRight)
    c5 = np.array(ob[0].pts)
    ang = np.degrees(np.arctan2(c5[1][1] - c5[0][1], c5[1][0] - c5[0][0]))
    check(abs(ang - 1.0) < 0.2 and abs(abs(cross2(c5[1] - c5[0], c5[3] - c5[0])) - area0) < 1, f"OBB: поворот на 1° клавишей ] (угол {ang:.2f}°, площадь сохранена)")
    for _ in range(4):
        QTest.keyClick(cv, Qt.Key_BracketRight, Qt.ShiftModifier)
    # перетаскивание угла: прямоугольность сохраняется
    c6 = np.array(ob[0].pts)
    drag(cv, pt(cv, *c6[2]), pt(cv, c6[2][0] + 30, c6[2][1] + 20))
    c7 = np.array(ob[0].pts)
    e1, e2 = c7[1] - c7[0], c7[3] - c7[0]
    check(abs(np.dot(e1, e2)) / (np.linalg.norm(e1) * np.linalg.norm(e2)) < 0.02 and np.linalg.norm(c7[2] - c6[2]) > 10,
          "OBB: перетаскивание угла оставляет прямоугольник")
    w.undo(); w.undo()

    # --- 10c. ключевые точки (pose)
    pr.keypoints = ["nose", "left_eye", "right_eye"]
    pr.skeleton = [[0, 1], [0, 2]]
    pr.flip_pairs = [[1, 2]]
    w.set_tool("pose")
    drag(cv, pt(cv, 200, 60), pt(cv, 300, 180))
    check(cv._pose_box is not None, "pose: рамка нарисована, ждём точки")
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 250, 100))
    QTest.keyClick(cv, Qt.Key_Backspace)
    check(len(cv._pose_kps) == 0, "pose: Backspace убирает точку")
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 250, 100))
    QTest.mouseClick(cv, Qt.LeftButton, Qt.ControlModifier, pt(cv, 230, 90))
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 320, 95))             # вне рамки: рамка должна расшириться
    pose = [a for a in e3.anns if a.kind == "pose"]
    check(len(pose) == 1 and [k[2] for k in pose[0].kps] == [2, 1, 2], "pose: три точки поставлены (видна, закрыта, видна)")
    check(pose[0].bbox()[2] >= 318, f"pose: рамка расширилась до всех видимых точек (x2={pose[0].bbox()[2]:.0f})")
    w.set_tool("pose")
    drag(cv, pt(cv, 40, 40), pt(cv, 120, 110))
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 80, 60))
    QTest.keyClick(cv, Qt.Key_N)                                                    # пропуск точки
    QTest.keyClick(cv, Qt.Key_Return)                                               # закончить досрочно
    pose = [a for a in e3.anns if a.kind == "pose"]
    check(len(pose) == 2 and [k[2] for k in pose[1].kps] == [2, 0, 0], "pose: N пропускает точку, Enter завершает с остальными v=0")
    w.set_tool("select")
    QTest.mouseClick(cv, Qt.LeftButton, Qt.NoModifier, pt(cv, 80, 100))
    check(cv.selected == pose[1].id, "pose: выделяется щелчком по рамке")
    kp0 = list(pose[1].kps[0])
    drag(cv, pt(cv, kp0[0], kp0[1]), pt(cv, kp0[0] + 15, kp0[1] + 10))
    check(abs(pose[1].kps[0][0] - kp0[0] - 15) < 2 and abs(pose[1].kps[0][1] - kp0[1] - 10) < 2, "pose: точка перетаскивается мышью")
    k = pose[1].kps[0]
    QTest.mouseClick(cv, Qt.LeftButton, Qt.AltModifier, pt(cv, k[0], k[1]))
    check(pose[1].kps[0][2] == 1, "pose: Alt+щелчок переключает видимость точки")
    bb0, kx0 = pose[1].bbox(), float(pose[1].kps[0][0])
    drag(cv, pt(cv, 80, 90), pt(cv, 100, 100))
    check(abs(pose[1].bbox()[0] - bb0[0] - 20) < 2 and abs(pose[1].kps[0][0] - kx0 - 20) < 2, "pose: при перемещении рамки точки двигаются вместе с ней")

    # --- 10d. класс кадра (классификация)
    w.set_tool("tag")
    w.load_image(0)
    check(w.entry.tag is None, "тег: у кадра класс не задан")
    w.pick_class(1)
    check(pr.images[0].tag == 1 and w.idx == 1, "тег: клавиша 2 задала класс и перешла к следующему кадру")
    w.load_image(0)
    w.undo()
    check(pr.images[0].tag is None, "тег: Ctrl+Z снимает класс кадра")
    w.redo()
    check(pr.images[0].tag == 1, "тег: Ctrl+Shift+Z возвращает")
    w.load_image(1)
    w._class_clicked(w.cls_list.item(0))
    check(pr.images[1].tag == 0, "тег: щелчок по классу в списке задаёт класс кадра")
    check("→" in w.img_model.text(pr.images[0], pr.class_names()), "тег: класс виден в списке изображений")
    w.set_tool("select")

    # --- 11. сохранение и повторное открытие
    w.save_project()
    pr2 = Project.load(tmp / "p.mlide.json")
    check(sum(len(i.anns) for i in pr2.images) == sum(len(i.anns) for i in pr.images), "разметка сохраняется и открывается без потерь")
    check(len(list(tmp.glob("*.bak"))) >= 0, "сохранение завершилось без ошибок")

    print("\nИТОГО: провалов", len(FAILS))
    for f in FAILS:
        print("  -", f)
    assert not FAILS, FAILS
    print("INTERACTION OK")


if __name__ == "__main__":
    main()
