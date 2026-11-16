"""Dialogs: augmentation presets, export, auto-annotate, stats."""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QListWidgetItem, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget, QScrollArea, QFrame)

from .. import autolabel, imgio
from ..augment import BUILTIN_PRESETS, FIELD_UI, AugParams, augment_sample, get_all_presets
from ..formats import FORMATS, MODEL_FAMILIES, TASK_OF, Cancelled, ExportOptions, export_dataset
from ..model import Annotation, Project
from .. import knowledge
from . import icons, style


def add_tips(root, mapping):
    """Подсказки при наведении: на сам виджет и на его подпись в форме."""
    from PySide6.QtWidgets import QFormLayout
    forms = root.findChildren(QFormLayout)
    for w, text in mapping.items():
        html_text = f"<p>{text}</p>"
        w.setToolTip(html_text)
        for fl in forms:
            lab = fl.labelForField(w)
            if lab is not None:
                lab.setToolTip(html_text)


def bgr_to_qpixmap(img):
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


def draw_anns(img, anns, colors, names):
    out = img.copy()
    for cls, kind, pts in anns:
        col = colors[cls % len(colors)] if colors else (0, 255, 0)
        P = np.asarray(pts, dtype=np.int32)
        if kind == "box":
            x1, y1 = P.min(0)
            x2, y2 = P.max(0)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            tl = (x1, max(12, y1 - 4))
        else:
            cv2.polylines(out, [P], True, col, 2)
            tl = tuple(P[P[:, 1].argmin()])
        if names and 0 <= cls < len(names):
            cv2.putText(out, names[cls], tl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


def hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


# =============================================================================== presets
class PresetDialog(QDialog):
    """Edit augmentation presets with a live random preview on the current image."""

    def __init__(self, project: Project, entry, parent=None, current: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Пресеты аугментаций")
        self.resize(980, 640)
        self.project, self.entry = project, entry
        self.rng = np.random.default_rng()
        self._loading = False
        self.base_img = None
        if entry:
            try:
                img = imgio.imread(entry.path)
                s = 640 / max(img.shape[:2])
                self.scale = min(1.0, s)
                self.base_img = cv2.resize(img, None, fx=self.scale, fy=self.scale) if s < 1 else img
            except Exception:
                pass

        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.lst = QListWidget()
        self.lst.setMaximumWidth(210)
        left.addWidget(QLabel("Пресеты"))
        left.addWidget(self.lst, 1)
        for text, fn in (("Дублировать как новый…", self.duplicate), ("Удалить", self.delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            left.addWidget(b)
        lay.addLayout(left)

        mid = QVBoxLayout()
        box = QGroupBox("Параметры")
        form = QFormLayout(box)
        self.spins: dict[str, QDoubleSpinBox] = {}
        for key, (label, lo, hi, step, dec, tip) in FIELD_UI.items():
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            sp.setToolTip(tip)
            sp.valueChanged.connect(self._edited)
            self.spins[key] = sp
            form.addRow(label, sp)
            form.labelForField(sp).setToolTip(f"<p>{tip}</p>")
            sp.setToolTip(f"<p>{tip}</p>")
        mid.addWidget(box)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        mid.addWidget(self.note)
        row = QHBoxLayout()
        self.save_btn = QPushButton("Сохранить изменения")
        self.save_btn.clicked.connect(self.save)
        row.addWidget(self.save_btn)
        mid.addLayout(row)
        lay.addLayout(mid)

        right = QVBoxLayout()
        right.addWidget(QLabel("Предпросмотр (текущее изображение, случайный результат)"))
        self.preview = QLabel("Сначала откройте изображение" if self.base_img is None else "")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(380, 380)
        self.preview.setStyleSheet(f"background:{style.CHART};border-radius:6px")
        right.addWidget(self.preview, 1)
        nb = QPushButton("Другой вариант")
        nb.clicked.connect(self.refresh_preview)
        right.addWidget(nb)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.accept)
        right.addWidget(close)
        lay.addLayout(right, 1)

        self.lst.currentItemChanged.connect(lambda cur, _p: self._select(cur.data(Qt.UserRole) if cur else None))
        self._reload_list(current)

    # ---- helpers
    def _all(self):
        return get_all_presets(self.project.presets)

    def _reload_list(self, select=None):
        self.lst.blockSignals(True)
        self.lst.clear()
        for k in self._all():
            it = QListWidgetItem(k)
            it.setData(Qt.UserRole, k)
            self.lst.addItem(it)
        self.lst.blockSignals(False)
        names = list(self._all())
        self.lst.setCurrentRow(names.index(select) if select in names else 2)
        self._select(self.lst.currentItem().data(Qt.UserRole))

    @property
    def current_name(self):
        it = self.lst.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _select(self, name):
        if not name:
            return
        p = self._all()[name]
        self._loading = True
        for k, sp in self.spins.items():
            sp.setValue(float(getattr(p, k)))
        self._loading = False
        builtin = name in BUILTIN_PRESETS
        self.save_btn.setEnabled(not builtin)
        for sp in self.spins.values():
            sp.setEnabled(not builtin)
        self.note.setText("Встроенный пресет (только чтение). Нажмите «Дублировать как новый…», чтобы изменить." if builtin
                          else "Свой пресет - измените значения и нажмите «Сохранить изменения».")
        self.refresh_preview()

    def params(self) -> AugParams:
        d = {k: (int(sp.value()) if k == "copies" else sp.value()) for k, sp in self.spins.items()}
        return AugParams.from_dict(d)

    def _edited(self):
        if not self._loading:
            self.refresh_preview()

    def save(self):
        n = self.current_name
        if n and n not in BUILTIN_PRESETS:
            self.project.presets[n] = self.params().to_dict()
            self.project.dirty = True

    def duplicate(self):
        name, ok = QInputDialog.getText(self, "Новый пресет", "Название:", text=f"{self.current_name} (копия)")
        if ok and name.strip():
            name = name.strip()
            self.project.presets[name] = self.params().to_dict()
            self.project.dirty = True
            self._reload_list(name)

    def delete(self):
        n = self.current_name
        if n in self.project.presets:
            del self.project.presets[n]
            self.project.dirty = True
            self._reload_list()

    def refresh_preview(self):
        if self.base_img is None or not self.entry:
            return
        trip = [(a.cls, a.kind, [[x * self.scale, y * self.scale] for x, y in a.pts]) for a in self.entry.anns]
        out, anns = augment_sample(self.base_img, trip, self.params(), self.rng)
        cols = [hex_to_bgr(c["color"]) for c in self.project.classes] or [(0, 255, 0)]
        out = draw_anns(out, anns, cols, self.project.class_names())
        pm = bgr_to_qpixmap(out).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "lst") and self.lst.currentItem():
            self.refresh_preview()


# =============================================================================== export
class ExportWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, project, opts):
        super().__init__()
        self.project, self.opts, self._cancel = project, opts, False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            msg = export_dataset(self.project, self.opts,
                                 lambda i, n, m: (self.progress.emit(i, n, m), not self._cancel)[1])
            self.done.emit(msg)
        except Cancelled:
            self.failed.emit("Отменено.")
        except Exception as e:  # noqa
            self.failed.emit(f"{type(e).__name__}: {e}")


class ExportDialog(QDialog):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Экспорт датасета")
        self.project = project
        s = project.export
        # содержимое прокручивается (окно не всегда помещается на экране), кнопки снизу всегда видны
        outer = QVBoxLayout(self)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 6, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        scr = self.screen().availableGeometry() if self.screen() else None
        self.resize(640, min(820, int(scr.height() * 0.85)) if scr else 640)
        adv = QPushButton("Подобрать настройки под мой датасет…")
        adv.setIcon(icons.icon("advisor"))
        adv.setToolTip("Советник проанализирует разметку и предложит модель, эпохи, размер изображения и пресет.")
        adv.clicked.connect(self._advise)
        lay.addWidget(adv)

        g = QGroupBox("Формат")
        f = QFormLayout(g)
        self.fmt = QComboBox()
        for k, v in FORMATS.items():
            self.fmt.addItem(v, k)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(s.get("fmt", "ultra_det"))))
        f.addRow(self.fmt)
        out_row = QHBoxLayout()
        self.out = QLineEdit(s.get("out_dir", ""))
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        out_row.addWidget(self.out, 1)
        out_row.addWidget(b)
        f.addRow("Папка экспорта", out_row)
        lay.addWidget(g)

        g = QGroupBox("Разделение выборки")
        f = QFormLayout(g)
        self.sp = {}
        for k, dv in (("train", 80), ("val", 20), ("test", 0)):
            w = QSpinBox()
            w.setRange(0, 100)
            w.setSuffix(" %")
            w.setValue(int(s.get(f"{k}_pct", dv)))
            self.sp[k] = w
            f.addRow({"train": "Обучение (train)", "val": "Валидация (val)", "test": "Тест (test)"}[k], w)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(int(s.get("seed", 0)))
        f.addRow("Seed (воспроизводимость)", self.seed)
        self.unl = QCheckBox("Добавить неразмеченные изображения как фон (пустые метки)")
        self.unl.setChecked(bool(s.get("include_unlabeled", False)))
        self.rev = QCheckBox("Только изображения, отмеченные проверенными")
        self.rev.setChecked(bool(s.get("only_reviewed", False)))
        f.addRow(self.unl)
        f.addRow(self.rev)
        self.group = QCheckBox("Не разделять похожие кадры между train/val/test (защита от утечки)")
        self.group.setChecked(bool(s.get("group_split", False)))
        self.group_hint = QLabel("")
        self.group_hint.setWordWrap(True)
        f.addRow(self.group)
        f.addRow(self.group_hint)
        try:
            from ..dedupe import source_key
            keys = [source_key(e.path) for e in project.images]
            if keys and len(keys) / max(1, len(set(keys))) > 3 and "group_split" not in s:
                self.group.setChecked(True)
            if keys and len(keys) / max(1, len(set(keys))) > 3:
                self.group_hint.setText("<span style='color:#f0a030'>Похоже, часть кадров - из одного видео/серии (имена с номерами). "
                                        "Рекомендуем оставить галочку включённой.</span>")
        except Exception:
            pass
        lay.addWidget(g)

        g = QGroupBox("Аугментация")
        f = QFormLayout(g)
        self.preset = QComboBox()
        for k in get_all_presets(project.presets):
            self.preset.addItem(k, k)
        legacy = {"None": "Без аугментаций", "Light": "Лёгкая", "Medium (YOLO default)": "Средняя (по умолчанию YOLO)",
                  "Heavy": "Сильная", "Aerial / drone": "Аэросъёмка / дрон", "Low light / night": "Тёмное время / ночь"}
        pname = s.get("preset", "Средняя (по умолчанию YOLO)")
        self.preset.setCurrentIndex(max(0, self.preset.findData(legacy.get(pname, pname))))
        self.bake = QCheckBox("Добавить аугментированные копии в train (офлайн)")
        self.bake.setChecked(bool(s.get("bake_aug", False)))
        self.online = QCheckBox("Записать этот пресет в train.py (онлайн, Ultralytics)")
        self.online.setChecked(bool(s.get("online_aug", True)))
        self.bake.setToolTip("Создаёт N дополнительных аугментированных изображений с метками на каждое обучающее (см. «копий» в пресете).")
        f.addRow("Пресет", self.preset)
        f.addRow(self.bake)
        f.addRow(self.online)
        lay.addWidget(g)

        g = QGroupBox("Скрипт обучения (только форматы Ultralytics)")
        f = QFormLayout(g)
        self.script = QCheckBox("Создать train.py")
        self.script.setChecked(bool(s.get("script", True)))
        self.family = QComboBox()
        self.family.addItems(list(MODEL_FAMILIES))
        self.family.setCurrentText(s.get("family", "YOLO11"))
        self.size = QComboBox()
        self.size.addItems(["n", "s", "m", "l", "x"])
        self.size.setCurrentText(s.get("size", "n"))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.epochs.setValue(int(s.get("epochs", 100)))
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(int(s.get("imgsz", 640)))
        self.batch = QSpinBox()
        self.batch.setRange(-1, 1024)
        self.batch.setValue(int(s.get("batch", 16)))
        f.addRow(self.script)
        f.addRow("Семейство модели", self.family)
        f.addRow("Размер модели", self.size)
        f.addRow("Эпохи", self.epochs)
        f.addRow("Размер изображения", self.imgsz)
        f.addRow("Batch (-1 = авто)", self.batch)
        self.freeze = QSpinBox()
        self.freeze.setRange(0, 30)
        self.freeze.setValue(int(s.get("freeze", 0)))
        self.freeze.setToolTip("Заморозить первые N слоёв предобученной модели (10 = backbone). Помогает на малых данных.")
        self.lr0 = QDoubleSpinBox()
        self.lr0.setDecimals(4)
        self.lr0.setRange(0, 0.1)
        self.lr0.setSingleStep(0.0005)
        self.lr0.setValue(float(s.get("lr0", 0)))
        self.lr0.setToolTip("Шаг обучения. 0 = по умолчанию YOLO. Для дообучения на малых данных ~0.001 (включит AdamW).")
        self.patience = QSpinBox()
        self.patience.setRange(1, 1000)
        self.patience.setValue(int(s.get("patience", 100)))
        f.addRow("Заморозка слоёв (0 = нет)", self.freeze)
        f.addRow("Шаг обучения lr0 (0 = авто)", self.lr0)
        f.addRow("Patience (ранняя остановка)", self.patience)
        lay.addWidget(g)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        outer.addWidget(self.bar)
        outer.addWidget(self.status)
        self.btns = QDialogButtonBox()
        self.go = self.btns.addButton("Экспортировать", QDialogButtonBox.AcceptRole)
        self.btns.addButton(QDialogButtonBox.Close)
        self.btns.accepted.connect(self._run)
        self.btns.rejected.connect(self.reject)
        outer.addWidget(self.btns)
        self.fmt.currentIndexChanged.connect(self._fmt_changed)
        self._fmt_changed()
        self.worker = None
        add_tips(self, {
            self.fmt: "В каком формате сохранить датасет. Для YOLOv5/v8/v9/v10/v11/v12/26 выбирайте Ultralytics: формат меток у них одинаковый. Detection - рамки, Segmentation - контуры, OBB - повёрнутые рамки.",
            self.out: "Папка, куда будет записан датасет (images/, labels/, data.yaml, train.py). Лучше выбирать пустую.",
            self.sp["train"]: "Доля картинок, на которых модель учится. Обычно 70-80%.",
            self.sp["val"]: "Доля картинок для проверки во время обучения: по ним строятся графики и выбирается лучшая эпоха. Обычно 10-20%, не меньше 20-30 картинок.",
            self.sp["test"]: "Отложенные картинки для финальной честной оценки: модель их не видит при обучении. 0 - не нужны (если данных мало).",
            self.seed: "Число, от которого зависит случайное перемешивание. Тот же seed - то же разбиение (воспроизводимость).",
            self.unl: "Неразмеченные картинки попадут в датасет с пустыми метками как «фон». Это уменьшает ложные срабатывания, но только если на них точно нет объектов. До ~10% датасета.",
            self.rev: "Экспортировать только изображения, отмеченные проверенными (клавиша R).",
            self.group: "Одинаковые и почти одинаковые кадры (дубли, соседние кадры видео) попадут только в одну часть выборки. Иначе модель «видит» копии проверочных картинок при обучении, и метрики получаются завышенными. Поиск похожих делается при экспорте и может занять время на больших датасетах.",
            self.preset: "Набор искажений для обучения (повороты, яркость, шум…). Чем меньше данных, тем сильнее нужна аугментация. Подробнее: Инструменты → Пресеты аугментаций.",
            self.bake: "Заранее создать N аугментированных копий каждой обучающей картинки и добавить их в датасет. Нужно для форматов без онлайн-аугментации (COCO, Darknet). Для Ultralytics обычно не нужно: она аугментирует на лету.",
            self.online: "Записать параметры пресета в train.py: Ultralytics применит искажения на лету в каждой эпохе. Не включайте вместе с офлайн-копиями без нужды - искажения удвоятся.",
            self.script: "Создать готовый train.py, который запускает обучение одной командой.",
            self.family: "Семейство YOLO: новее - обычно точнее и быстрее. YOLO11 - хороший выбор по умолчанию. Формат меток у всех одинаковый.",
            self.size: "Размер модели: n (nano) - самая быстрая, x - самая точная, но тяжёлая и требует много данных. На малых данных берите n или s.",
            self.epochs: "Сколько раз модель пройдёт по всем обучающим картинкам. Типично 50-300. Смотрите на график: когда качество перестало расти, дальше учить не нужно.",
            self.imgsz: "Размер картинки при обучении (по длинной стороне). 640 - стандарт; для мелких объектов 960-1280 (медленнее, больше памяти).",
            self.batch: "Сколько картинок обрабатывается за раз. Больше - стабильнее, но нужна видеопамять. При ошибке out of memory уменьшите. -1 - подобрать автоматически.",
            self.freeze: "Заморозить первые N слоёв предобученной модели (10 = основная часть). Помогает на малых данных: модель не «портит» уже выученное.",
            self.lr0: "Шаг обучения - насколько сильно модель правит себя за раз. 0 - значение YOLO по умолчанию. Для дообучения на малых данных ~0.001 (включится AdamW). Слишком большой - обучение «скачет».",
            self.patience: "Сколько эпох подряд без улучшения терпеть, прежде чем остановить обучение автоматически.",
        })

    def _advise(self):
        from .learn import AdvisorDialog
        dlg = AdvisorDialog(self.project, self)
        if dlg.exec() == QDialog.Accepted and dlg.settings:
            s = dlg.settings
            self.preset.setCurrentIndex(max(0, self.preset.findData(s["preset"])))
            self.epochs.setValue(s["epochs"])
            self.imgsz.setValue(s["imgsz"])
            self.size.setCurrentText(s["size"])
            self.batch.setValue(s["batch"])
            self.bake.setChecked(s["bake_aug"])
            self.online.setChecked(s["online_aug"])
            self.freeze.setValue(s.get("freeze", 0))
            self.lr0.setValue(s.get("lr0", 0))
            self.patience.setValue(s.get("patience", 100))
            for k in ("train", "val", "test"):
                self.sp[k].setValue(int(s[f"{k}_pct"]))

    def _fmt_changed(self):
        ultra = self.fmt.currentData() in TASK_OF
        if self.fmt.currentData() == "ultra_cls" and self.imgsz.value() == 640:
            self.imgsz.setValue(224)
        elif self.fmt.currentData() != "ultra_cls" and self.imgsz.value() == 224:
            self.imgsz.setValue(640)
        for w in (self.script, self.family, self.size, self.epochs, self.imgsz, self.batch, self.online,
                  self.freeze, self.lr0, self.patience):
            w.setEnabled(ultra)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Папка экспорта", self.out.text() or str(Path.home()))
        if d:
            self.out.setText(d)

    def _run(self):
        if not self.out.text().strip():
            QMessageBox.warning(self, "Экспорт", "Выберите папку экспорта.")
            return
        out = Path(self.out.text().strip())
        if out.exists() and any(out.iterdir()):
            r = QMessageBox.question(self, "Экспорт", f"Папка '{out}' не пуста. Прежний экспорт в ней (папки с "
                                     "изображениями и метками) будет удалён и заменён новым; результаты обучения "
                                     "в runs/ сохранятся. Продолжить?")
            if r != QMessageBox.Yes:
                return
        presets = get_all_presets(self.project.presets)
        o = ExportOptions(
            fmt=self.fmt.currentData(), out_dir=str(out),
            train=self.sp["train"].value(), val=self.sp["val"].value(), test=self.sp["test"].value(),
            seed=self.seed.value(), include_unlabeled=self.unl.isChecked(), only_reviewed=self.rev.isChecked(),
            bake_aug=self.bake.isChecked(), aug=presets[self.preset.currentData()],
            write_train_script=self.script.isChecked(), family=self.family.currentText(),
            size=self.size.currentText(), epochs=self.epochs.value(), imgsz=self.imgsz.value(),
            batch=self.batch.value(), online_aug=self.online.isChecked(),
            freeze=self.freeze.value(), lr0=self.lr0.value(), patience=self.patience.value(),
            group_split=self.group.isChecked())
        self.project.export = {
            "fmt": o.fmt, "out_dir": o.out_dir, "train_pct": o.train, "val_pct": o.val, "test_pct": o.test,
            "seed": o.seed, "include_unlabeled": o.include_unlabeled, "only_reviewed": o.only_reviewed,
            "preset": self.preset.currentData(), "bake_aug": o.bake_aug, "online_aug": o.online_aug,
            "script": o.write_train_script, "family": o.family, "size": o.size, "epochs": o.epochs,
            "imgsz": o.imgsz, "batch": o.batch, "freeze": o.freeze, "lr0": o.lr0, "patience": o.patience, "group_split": o.group_split}
        self.project.dirty = True
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.worker = ExportWorker(self.project, o)
        self.worker.progress.connect(lambda i, n, m: (self.bar.setMaximum(n), self.bar.setValue(i),
                                                      self.status.setText(m)))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _done(self, msg):
        self.bar.setValue(self.bar.maximum())
        self.status.setText(msg)
        self.go.setEnabled(True)

    def _failed(self, msg):
        self.status.setText(f"<span style='color:#ff6b6b'>{msg}</span>")
        self.go.setEnabled(True)

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().reject()


# =============================================================================== auto-annotate
class AutoWorker(QThread):
    progress = Signal(int, int)
    done = Signal(int, int)       # images processed, boxes added
    failed = Signal(str)

    def __init__(self, project, entries, model_path, conf, iou, imgsz, name_map, create_missing, replace_suggested):
        super().__init__()
        self.project, self.entries = project, entries
        self.model_path, self.conf, self.iou, self.imgsz = model_path, conf, iou, imgsz
        self.name_map, self.create_missing, self.replace = name_map, create_missing, replace_suggested
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            model = autolabel.load_model(self.model_path, self.imgsz)
        except Exception as e:  # noqa
            self.failed.emit(f"Не удалось загрузить модель:\n{e}")
            return
        added = 0
        n = len(self.entries)
        try:
            for i, e in enumerate(self.entries):
                if self._cancel:
                    break
                self.progress.emit(i, n)
                img = imgio.imread(e.path)
                dets = model.predict(img, self.conf, self.iou)
                if self.replace:
                    e.anns = [a for a in e.anns if not a.suggested]
                for d in dets:
                    cls = self._map(d, model)
                    if cls is None:
                        continue
                    if d.poly:
                        e.anns.append(Annotation(cls, "poly", d.poly, suggested=True, conf=d.conf))
                    else:
                        x1, y1, x2, y2 = d.box
                        e.anns.append(Annotation(cls, "box", [[x1, y1], [x2, y2]], suggested=True, conf=d.conf))
                    added += 1
            self.done.emit(min(i + 1, n) if n else 0, added)
        except Exception as ex:  # noqa
            self.failed.emit(f"{type(ex).__name__}: {ex}")

    def _map(self, d, model):
        names = self.project.class_names()
        if self.name_map:                          # user-supplied model class names
            if d.cls >= len(self.name_map):
                return None
            nm = self.name_map[d.cls]
        elif d.name:                                # names embedded in .pt
            nm = d.name
        else:                                       # fall back to index
            return d.cls if d.cls < len(names) else None
        if nm in names:
            return names.index(nm)
        return self.project.add_class(nm) if self.create_missing else None


class AutoLabelDialog(QDialog):
    def __init__(self, project: Project, current_entry, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Авторазметка с помощью модели YOLO")
        self.resize(560, 470)
        self.project, self.current = project, current_entry
        self.worker = None
        lay = QVBoxLayout(self)
        info = QLabel("Загрузите свою обученную (или любую YOLO) модель, и она предложит рамки. Они появятся "
                      "пунктиром как <b>подсказки</b> - проверьте их, затем примите (Y) или удалите.<br>"
                      "<b>.onnx</b>: работает сразу (детекция YOLOv5/v7/v8/v9/v10/v11/v12/26). "
                      "<b>.pt</b>: нужен <code>pip install ultralytics</code> (даёт и контуры сегментации).")
        info.setWordWrap(True)
        lay.addWidget(info)
        f = QFormLayout()
        row = QHBoxLayout()
        self.path = QLineEdit(project.export.get("auto_model", ""))
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(b)
        f.addRow("Модель", row)
        self.names = QLineEdit()
        self.names.setPlaceholderText("необязательно: названия классов модели через запятую (по умолчанию - по номеру/имени)")
        f.addRow("Классы модели", self.names)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.05, 0.99)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.45)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        f.addRow("Порог уверенности", self.conf)
        f.addRow("NMS IoU (подавление дублей)", self.iou)
        f.addRow("Размер изображения", self.imgsz)
        self.scope = QComboBox()
        self.scope.addItems(["Текущему изображению", "Всем неразмеченным", "Всем изображениям"])
        self.scope.setCurrentIndex(1)
        f.addRow("Применить к", self.scope)
        self.newcls = QCheckBox("Создавать недостающие классы в проекте")
        self.repl = QCheckBox("Заменять прежние непроверенные подсказки")
        self.repl.setChecked(True)
        f.addRow(self.newcls)
        f.addRow(self.repl)
        lay.addLayout(f)
        self.bar = QProgressBar()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        self.btns = QDialogButtonBox()
        self.run_btn = self.btns.addButton("Запустить", QDialogButtonBox.AcceptRole)
        self.btns.addButton(QDialogButtonBox.Close)
        self.btns.accepted.connect(self._run)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)
        add_tips(self, {
            self.path: "Файл обученной модели: .onnx (работает без torch) или .pt (нужен пакет ultralytics).",
            self.names: "Если классы модели называются иначе или идут в другом порядке - перечислите их через запятую в порядке модели. Они сопоставятся с классами проекта по имени.",
            self.conf: "Порог уверенности: показываются находки, в которых модель уверена не меньше этого значения. Ниже - больше находок, но и больше ошибок. 0.25 удобно для подсказок.",
            self.iou: "Порог слияния дублей (NMS): если две рамки одного класса пересекаются сильнее этого значения, остаётся лучшая. Для плотно стоящих объектов повышайте (0.6-0.7).",
            self.imgsz: "До какого размера уменьшается картинка перед подачей в модель. Должен совпадать с размером, на котором модель обучали (обычно 640).",
            self.scope: "Какие изображения обработать: только текущее, все неразмеченные или все.",
            self.newcls: "Если модель знает класс, которого нет в проекте, создать его автоматически.",
            self.repl: "Перед запуском удалить прежние непринятые подсказки, чтобы не было дублей.",
        })

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Модель", self.path.text(), "Модели YOLO (*.onnx *.pt)")
        if p:
            self.path.setText(p)

    def _run(self):
        mp = self.path.text().strip()
        if not Path(mp).exists():
            QMessageBox.warning(self, "Авторазметка", "Выберите файл модели (.onnx или .pt).")
            return
        sc = self.scope.currentIndex()
        if sc == 0:
            entries = [self.current] if self.current else []
        elif sc == 1:
            entries = [e for e in self.project.images if not [a for a in e.anns if not a.suggested]]
        else:
            entries = list(self.project.images)
        if not entries:
            self.status.setText("Нечего обрабатывать.")
            return
        names = [n.strip() for n in self.names.text().split(",") if n.strip()]
        self.project.export["auto_model"] = mp
        self.run_btn.setEnabled(False)
        self.bar.setMaximum(len(entries))
        self.worker = AutoWorker(self.project, entries, mp, self.conf.value(), self.iou.value(), self.imgsz.value(),
                                 names, self.newcls.isChecked(), self.repl.isChecked())
        self.worker.progress.connect(lambda i, n: (self.bar.setValue(i), self.status.setText(f"{i}/{n} …")))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _done(self, n_img, n_box):
        self.project.dirty = True
        self.bar.setValue(self.bar.maximum())
        self.status.setText(f"Готово: подсказок {n_box}, изображений {n_img}. Закройте окно и проверьте их.")
        self.run_btn.setEnabled(True)
        self.accepted_changes = True

    def _failed(self, msg):
        self.status.setText(f"<span style='color:#ff6b6b'>{msg}</span>")
        self.run_btn.setEnabled(True)

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().reject()


# =============================================================================== stats
def dataset_report(project: Project) -> str:
    names = project.class_names()
    n = len(project.images)
    labeled = [e for e in project.images if [a for a in e.anns if not a.suggested] or e.tag is not None]
    counts = [0] * max(1, len(names))
    tiny, dup, oob = [], [], []
    areas = []
    for e in project.images:
        seen = set()
        if e.tag is not None and 0 <= e.tag < len(counts):
            counts[e.tag] += 1
        for a in e.anns:
            if a.suggested:
                continue
            if 0 <= a.cls < len(counts):
                counts[a.cls] += 1
            x1, y1, x2, y2 = a.bbox()
            areas.append(((x2 - x1) * (y2 - y1)) / (e.w * e.h))
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                tiny.append(e.name)
            if x1 < -1 or y1 < -1 or x2 > e.w + 1 or y2 > e.h + 1:
                oob.append(e.name)
            key = (a.cls, round(x1), round(y1), round(x2), round(y2))
            if key in seen:
                dup.append(e.name)
            seen.add(key)
    sug = sum(1 for e in project.images for a in e.anns if a.suggested)
    lines = [f"Изображений: {n}   размечено: {len(labeled)}   без разметки: {n - len(labeled)}   "
             f"проверено: {sum(e.reviewed for e in project.images)}",
             f"Объектов: {sum(counts)}   подсказок на проверку: {sug}", "", "По классам:"]
    mx = max(counts) if counts else 1
    for i, nm in enumerate(names):
        bar = "█" * int(30 * counts[i] / mx) if mx else ""
        warn = "   (внимание: мало примеров, <30)" if counts[i] < 30 else ""
        lines.append(f"  {i:>2} {nm:<20} {counts[i]:>6} {bar}{warn}")
    if areas:
        a = np.array(areas)
        lines += ["", f"Размер объектов (доля площади кадра): медиана {np.median(a):.3f}, "
                      f"мелких (<1%): {100 * (a < 0.01).mean():.0f}%"]
        if (a < 0.01).mean() > 0.5:
            lines.append("  Много мелких объектов: увеличьте размер изображения (imgsz 960-1280) или обучайте по тайлам.")
    for title, lst in (("Крошечные рамки (<4 px)", tiny), ("Рамки за пределами кадра", oob), ("Дубликаты рамок", dup)):
        if lst:
            lines += ["", f"Внимание - {title}: {len(lst)}  например: " + ", ".join(sorted(set(lst))[:5])]
    if not (tiny or oob or dup):
        lines += ["", "Структурных проблем не найдено."]
    return "\n".join(lines)


class StatsDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Статистика и проверка датасета")
        self.resize(640, 460)
        lay = QVBoxLayout(self)
        t = QPlainTextEdit(dataset_report(project))
        t.setReadOnly(True)
        t.setStyleSheet("font-family: Consolas, monospace;")
        lay.addWidget(t)
        b = QDialogButtonBox(QDialogButtonBox.Close)
        b.rejected.connect(self.accept)
        lay.addWidget(b)
