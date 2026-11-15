"""Проверка разметки обученной моделью: поиск возможных ошибок и «что размечать дальше»."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import autolabel, imgio
from .dialogs import add_tips


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def analyze_labels(gt, preds, names, conf_hi=0.5):
    """gt: [(cls, box)], preds: [Detection]. Возвращает (score, [описания проблем])."""
    issues, score = [], 0.0
    nm = lambda c: names[c] if 0 <= c < len(names) else f"#{c}"
    used = set()
    for cls, box in gt:
        best, bi = 0.0, -1
        for j, d in enumerate(preds):
            if j in used:
                continue
            v = _iou(box, d.box)
            if v > best:
                best, bi = v, j
        if bi >= 0 and best >= 0.5:
            used.add(bi)
            d = preds[bi]
            if d.cls != cls and d.conf >= conf_hi:
                issues.append(f"класс «{nm(cls)}», модель уверена ({d.conf:.2f}): «{nm(d.cls)}»")
                score += 1.0 + d.conf
        elif best < 0.3:
            issues.append(f"рамка «{nm(cls)}»: модель там ничего не видит (возможно лишняя или неточная)")
            score += 0.6
        else:
            issues.append(f"рамка «{nm(cls)}»: плохо совпадает с моделью (IoU {best:.2f}) - проверьте границы")
            score += 0.4
    for j, d in enumerate(preds):
        if j in used or d.conf < conf_hi:
            continue
        if all(_iou(d.box, b) < 0.3 for _c, b in gt):
            issues.append(f"модель видит «{nm(d.cls)}» ({d.conf:.2f}), а в разметке пусто - возможно, объект пропущен")
            score += 1.0 + d.conf
    return score, issues


class _Worker(QThread):
    progress = Signal(int, int)
    row = Signal(int, float, str)         # индекс изображения, оценка, описание
    done = Signal()
    failed = Signal(str)

    def __init__(self, project, model_path, mode, conf, imgsz=640):
        super().__init__()
        self.project, self.model_path, self.mode, self.conf, self.imgsz = project, model_path, mode, conf, imgsz
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            model = autolabel.load_model(self.model_path, self.imgsz)
        except Exception as e:  # noqa
            self.failed.emit(f"Не удалось загрузить модель: {e}")
            return
        names = self.project.class_names()
        todo = []
        for i, e in enumerate(self.project.images):
            real = [a for a in e.anns if not a.suggested]
            if (self.mode == 0 and real) or (self.mode == 1 and not real):
                todo.append(i)
        try:
            for k, i in enumerate(todo):
                if self._cancel:
                    break
                self.progress.emit(k, len(todo))
                e = self.project.images[i]
                try:
                    img = imgio.imread(e.path)
                except Exception:
                    continue
                preds = model.predict(img, 0.10 if self.mode == 1 else self.conf, 0.45)
                if self.mode == 0:
                    gt = [(a.cls, a.bbox()) for a in e.anns if not a.suggested]
                    score, issues = analyze_labels(gt, [d for d in preds if d.conf >= self.conf], names)
                    if issues:
                        self.row.emit(i, score, "; ".join(issues[:3]) + (f" (+{len(issues) - 3})" if len(issues) > 3 else ""))
                else:
                    # самые поучительные кадры - где модель сомневается (уверенность около 0.5)
                    if not preds:
                        self.row.emit(i, 0.05, "модель ничего не нашла: либо пустой кадр, либо незнакомый объект")
                        continue
                    unc = [1 - abs(2 * d.conf - 1) for d in preds]
                    score = sum(unc) / len(unc) + 0.1 * len(preds)
                    mid = sum(1 for d in preds if 0.25 <= d.conf <= 0.65)
                    self.row.emit(i, score, f"находок: {len(preds)}, из них неуверенных: {mid}")
            self.done.emit()
        except Exception as ex:  # noqa
            self.failed.emit(f"{type(ex).__name__}: {ex}")


class ReviewDialog(QDialog):
    goto = Signal(int)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Проверка разметки моделью")
        self.resize(980, 620)
        self.project, self.worker = project, None
        lay = QVBoxLayout(self)
        intro = QLabel("Обученная модель просматривает изображения и подсказывает, куда смотреть в первую очередь. "
                       "Это <b>подсказки, а не приговор</b>: модель тоже ошибается, решение за вами.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        self.path = QLineEdit(project.export.get("auto_model", ""))
        self.path.setPlaceholderText("Модель .onnx / .pt")
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        row.addWidget(QLabel("Модель:"))
        row.addWidget(self.path, 1)
        row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["Найти ошибки в моей разметке", "Что размечать дальше (неуверенные кадры)"])
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.35)
        self.go = QPushButton("Запустить")
        self.go.setProperty("primary", True)
        self.go.clicked.connect(self._run)
        row.addWidget(QLabel("Режим:"))
        row.addWidget(self.mode, 1)
        row.addWidget(QLabel("Порог уверенности:"))
        row.addWidget(self.conf)
        row.addWidget(self.go)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = QLabel("")
        lay.addWidget(self.status)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Изображение", "Подозрительность", "Что заметила модель"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 230)
        self.table.setColumnWidth(1, 120)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._open)
        lay.addWidget(self.table, 1)
        lay.addWidget(QLabel("<span style='color:#8b92a1'>Двойной щелчок по строке открывает изображение в главном окне. "
                             "Окно можно оставить открытым и переключаться между кадрами.</span>"))
        add_tips(self, {
            self.mode: "«Найти ошибки»: сравнивает модель с вашей разметкой на размеченных кадрах (пропущенные объекты, не тот класс, неточные рамки). "
                       "«Что размечать дальше»: на неразмеченных кадрах ищет те, где модель неуверена - они полезнее всего для обучения.",
            self.conf: "Находки модели с уверенностью ниже этого порога игнорируются. Выше - меньше ложных подозрений, но и меньше найденных ошибок.",
            self.path: "Лучше использовать модель, обученную на этом же датасете (best.onnx из окна «Результаты»).",
        })

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Модель", self.path.text(), "YOLO (*.onnx *.pt)")
        if p:
            self.path.setText(p)

    def _run(self):
        mp = self.path.text().strip()
        if not Path(mp).exists():
            QMessageBox.warning(self, "Проверка разметки", "Укажите файл модели (.onnx или .pt).")
            return
        self.project.export["auto_model"] = mp
        self.table.setRowCount(0)
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.worker = _Worker(self.project, mp, self.mode.currentIndex(), self.conf.value())
        self.worker.progress.connect(lambda d, t: (self.bar.setRange(0, t), self.bar.setValue(d)))
        self.worker.row.connect(self._row)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(lambda m: (self.status.setText(f"<span style='color:#ff6b6b'>{m}</span>"), self.go.setEnabled(True),
                                              self.bar.setVisible(False)))
        self.worker.start()

    def _row(self, i, score, text):
        r = self.table.rowCount()
        self.table.insertRow(r)
        it0 = QTableWidgetItem(Path(self.project.images[i].path).name)
        it0.setData(Qt.UserRole, i)
        it1 = QTableWidgetItem(f"{score:.2f}")
        it1.setData(Qt.UserRole, score)
        self.table.setItem(r, 0, it0)
        self.table.setItem(r, 1, it1)
        self.table.setItem(r, 2, QTableWidgetItem(text))

    def _done(self):
        rows = []
        for r in range(self.table.rowCount()):
            rows.append((self.table.item(r, 1).data(Qt.UserRole), self.table.item(r, 0).data(Qt.UserRole),
                         self.table.item(r, 0).text(), self.table.item(r, 1).text(), self.table.item(r, 2).text()))
        rows.sort(key=lambda x: -x[0])
        self.table.setRowCount(0)
        for _s, i, name, sc, txt in rows[:2000]:
            r = self.table.rowCount()
            self.table.insertRow(r)
            it0 = QTableWidgetItem(name)
            it0.setData(Qt.UserRole, i)
            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, QTableWidgetItem(sc))
            self.table.setItem(r, 2, QTableWidgetItem(txt))
        self.go.setEnabled(True)
        self.bar.setVisible(False)
        self.status.setText(f"Найдено кадров для просмотра: {len(rows)}." if rows else "Подозрительных кадров не найдено ✔")

    def _open(self, row, _c):
        i = self.table.item(row, 0).data(Qt.UserRole)
        if i is not None:
            self.goto.emit(int(i))

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().reject()
