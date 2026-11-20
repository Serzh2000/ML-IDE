"""Мастер «Импорт разметки»: сам определяет формат (YOLO, COCO, VOC, LabelMe, CVAT, VIA, Label Studio, CSV/JSON-таблицы, файл-сопоставление),
показывает, сколько картинок найдено, даёт поправить столбцы таблицы и сопоставить классы, затем записывает разметку в проект."""
from __future__ import annotations

from . import winflags

import html
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton, QTableWidget, QTableWidgetItem,
                               QTextBrowser, QVBoxLayout)

from .. import annimport as AI
from .. import i18n
from . import icons

NEW_CLASS = "(как в источнике)"
SKIP_CLASS = "(пропустить)"
NONE = "(нет)"
BOX_FORMATS = [("xyxy", "углы: xmin, ymin, xmax, ymax"), ("xywh", "x, y, ширина, высота"), ("cxcywh", "центр x, y, ширина, высота")]


class _Analyze(QThread):
    done = Signal(object, list)
    failed = Signal(str)

    def __init__(self, paths, images_dir):
        super().__init__()
        self.paths, self.images_dir = paths, images_dir

    def run(self):
        try:
            ctx = AI.SourceContext(self.paths, self.images_dir)
            self.done.emit(ctx, AI.detect(ctx))
        except Exception as e:  # noqa
            self.failed.emit(f"{type(e).__name__}: {e}")


class _Parse(QThread):
    done = Signal(object, object)
    failed = Signal(str)

    def __init__(self, ctx, det, opt, project):
        super().__init__()
        self.ctx, self.det, self.opt, self.project = ctx, det, opt, project

    def run(self):
        try:
            parsed = AI.parse(self.ctx, self.det, self.opt)
            self.done.emit(parsed, AI.preview(self.ctx, parsed, self.project))
        except Exception as e:  # noqa
            self.failed.emit(f"{type(e).__name__}: {e}")


class _Apply(QThread):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, project, ctx, det, opt, parsed):
        super().__init__()
        self.project, self.ctx, self.det, self.opt, self.parsed = project, ctx, det, opt, parsed
        self.cancelled = False

    def run(self):
        try:
            rep = AI.run_import(self.project, self.ctx, self.det, self.opt,
                                lambda i, n, _r: (self.progress.emit(i, n), not self.cancelled)[1], parsed=self.parsed)
            self.done.emit(rep)
        except Exception as e:  # noqa
            self.failed.emit(f"{type(e).__name__}: {e}")


class ImportDialog(QDialog):
    def __init__(self, project, parent=None, paths=None, images_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Импорт разметки")
        self.resize(900, 760)
        self.project = project
        self.paths: list[str] = [str(p) for p in (paths or [])]
        self.images_dir: str | None = str(images_dir) if images_dir else None
        self.ctx = None
        self.dets: list = []
        self.parsed = None
        self.pv = None
        self.report = None
        self._thread = None
        self._busy = False
        self._table_header: list[str] = []
        self._lang = i18n.lang()

        lay = winflags.scroll_layout(self)
        intro = QLabel("Выберите папку или файлы с разметкой - формат определится сам: YOLO, COCO, Pascal VOC, LabelMe, CVAT, VIA, Label Studio, "
                       "CreateML, CSV / JSON-таблицы, список «картинка → файл разметки». Картинки находятся по именам и путям из файлов разметки.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        row = QHBoxLayout()
        self.src_edit = QLineEdit("; ".join(self.paths))
        self.src_edit.setReadOnly(True)
        self.src_edit.setPlaceholderText("Папка или файлы с разметкой…")
        b1 = QPushButton("Папка…")
        b1.setIcon(icons.icon("folder"))
        b1.clicked.connect(self._pick_folder)
        b2 = QPushButton("Файлы…")
        b2.setIcon(icons.icon("file"))
        b2.clicked.connect(self._pick_files)
        row.addWidget(self.src_edit, 1)
        row.addWidget(b1)
        row.addWidget(b2)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.img_edit = QLineEdit(self.images_dir or "")
        self.img_edit.setReadOnly(True)
        self.img_edit.setPlaceholderText("Папка с картинками (если лежит отдельно; иначе ищется рядом с разметкой)")
        b3 = QPushButton("Папка с картинками…")
        b3.clicked.connect(self._pick_images)
        b4 = QPushButton("Сбросить")
        b4.clicked.connect(self._clear_images)
        row.addWidget(self.img_edit, 1)
        row.addWidget(b3)
        row.addWidget(b4)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Формат:"))
        self.fmt = QComboBox()
        self.fmt.currentIndexChanged.connect(self._format_changed)
        row.addWidget(self.fmt, 1)
        lay.addLayout(row)

        # --- столбцы таблицы (только для таблиц)
        self.map_box = QGroupBox("Столбцы таблицы")
        f = QFormLayout(self.map_box)
        self.c_image = QComboBox()
        self.c_class = QComboBox()
        self.c_geo = QComboBox()
        for fmt, title in BOX_FORMATS:
            self.c_geo.addItem(title, fmt)
        self.c_geo.addItem("рамка в одном столбце (строка)", "str")
        self.c_geo.addItem("полигон в одном столбце", "poly")
        self.c_geo.addItem("без координат (класс всей картинки)", "none")
        self.c_cols = [QComboBox() for _ in range(4)]
        self.c_w = QComboBox()
        self.c_h = QComboBox()
        self.c_unit = QComboBox()
        for k, t in (("auto", "определить сам"), ("px", "пиксели"), ("norm", "доли кадра 0..1")):
            self.c_unit.addItem(t, k)
        f.addRow("Картинка:", self.c_image)
        f.addRow("Класс:", self.c_class)
        f.addRow("Координаты:", self.c_geo)
        rowc = QHBoxLayout()
        self._col_labels = []
        for c in self.c_cols:
            rowc.addWidget(c)
        f.addRow("Столбцы координат:", rowc)
        rowd = QHBoxLayout()
        rowd.addWidget(self.c_w)
        rowd.addWidget(self.c_h)
        f.addRow("Размер кадра (ширина, высота):", rowd)
        f.addRow("Единицы:", self.c_unit)
        for w in (self.c_image, self.c_class, self.c_geo, self.c_w, self.c_h, self.c_unit, *self.c_cols):
            w.currentIndexChanged.connect(self._mapping_changed)
        lay.addWidget(self.map_box)

        self.info = QTextBrowser()
        self.info.setMaximumHeight(170)
        lay.addWidget(self.info)

        lay.addWidget(QLabel("Классы: как назвать в проекте"))
        self.cls_table = QTableWidget(0, 3)
        self.cls_table.setHorizontalHeaderLabels(["Класс в источнике", "Объектов", "В проекте"])
        self.cls_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cls_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.cls_table.verticalHeader().setVisible(False)
        self.cls_table.setMinimumHeight(140)
        lay.addWidget(self.cls_table, 1)

        opts = QGroupBox("Как записать")
        ov = QVBoxLayout(opts)
        self.r_append = QRadioButton("Если у картинки уже есть разметка - дописать")
        self.r_replace = QRadioButton("Если у картинки уже есть разметка - заменить новой")
        self.r_skip = QRadioButton("Если у картинки уже есть разметка - пропустить её")
        self.r_append.setChecked(True)
        self.k_add = QCheckBox("Добавлять в проект картинки, которых там ещё нет")
        self.k_add.setChecked(True)
        self.k_tags = QCheckBox("Переносить класс всей картинки (классификация)")
        self.k_tags.setChecked(True)
        for w in (self.r_append, self.r_replace, self.r_skip, self.k_add, self.k_tags):
            ov.addWidget(w)
        lay.addWidget(opts)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        self.go = QPushButton("Импортировать")
        self.go.setProperty("primary", True)
        self.go.setEnabled(False)
        self.go.clicked.connect(self._run)
        row.addWidget(close)
        row.addWidget(self.go)
        lay.addLayout(row)
        self.map_box.setVisible(False)
        if self.paths:
            self._analyze()

    # ------------------------------------------------------------------ выбор источника
    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Папка с разметкой (или с датасетом целиком)")
        if d:
            self.paths = [d]
            self._after_pick()

    def _pick_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Файлы разметки", "",
                                             "Разметка (*.json *.jsonl *.csv *.tsv *.xml *.txt *.list *.yaml *.yml);;Все файлы (*)")
        if fs:
            self.paths = fs
            self._after_pick()

    def _pick_images(self):
        d = QFileDialog.getExistingDirectory(self, "Папка с картинками")
        if d:
            self.images_dir = d
            self.img_edit.setText(d)
            if self.paths:
                self._analyze()

    def _clear_images(self):
        self.images_dir = None
        self.img_edit.clear()
        if self.paths:
            self._analyze()

    def _after_pick(self):
        self.src_edit.setText("; ".join(self.paths))
        self._analyze()

    # ------------------------------------------------------------------ анализ
    def _busy_on(self, text):
        self._busy = True
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.status.setText(text)

    def _busy_off(self):
        self._busy = False
        self.bar.setVisible(False)
        if getattr(self, "_pending", False):                       # пока шёл разбор, пользователь сменил настройки: перечитываем
            self._pending = False
            QTimer.singleShot(0, self._reparse)

    def _analyze(self):
        self._busy_on("Определяю формат…")
        self.fmt.blockSignals(True)
        self.fmt.clear()
        self.fmt.blockSignals(False)
        self._thread = _Analyze(self.paths, self.images_dir)
        self._thread.done.connect(self._analyzed)
        self._thread.failed.connect(self._failed)
        self._thread.start()

    def _analyzed(self, ctx, dets):
        self._busy_off()
        self.ctx, self.dets = ctx, dets
        self.fmt.blockSignals(True)
        self.fmt.clear()
        for d in dets:
            self.fmt.addItem(f"{AI.title_of(d.parser, self._lang)} - {d.summary} ({int(d.score * 100)}%)", d)
        self.fmt.blockSignals(False)
        if not dets:
            self.info.setHtml("<p><b>Формат разметки не распознан.</b></p><p>Что можно попробовать: выбрать папку выше (датасет целиком) "
                              "или сам файл разметки; для таблиц нужен столбец с именем картинки и столбец с классом или координатами; "
                              "для YOLO - txt рядом с картинками или в папке labels.</p>")
            self.status.setText("")
            self.cls_table.setRowCount(0)
            self.map_box.setVisible(False)
            return
        self.fmt.setCurrentIndex(0)
        self._format_changed()

    def _failed(self, msg):
        self._busy_off()
        self.status.setText(f"<span style='color:#ff6b6b'>{html.escape(msg)}</span>")

    def current_det(self):
        i = self.fmt.currentIndex()
        return self.fmt.itemData(i) if i >= 0 else None

    # ------------------------------------------------------------------ формат и столбцы
    def _format_changed(self, *_):
        det = self.current_det()
        if det is None or self.ctx is None:
            return
        is_table = det.parser == "table"
        self.map_box.setVisible(is_table)
        if is_table:
            self._load_mapping_editor(det)
        self._reparse()

    def _load_mapping_editor(self, det):
        from ..annimport.parsers import table
        try:
            header, _rows, _hl = table.load_rows(self.ctx, Path(det.params["file"]), limit=300)
        except Exception:  # noqa
            header = []
        self._table_header = header
        m = det.params.get("mapping") or {}
        for w in (self.c_image, self.c_class, self.c_w, self.c_h, *self.c_cols):
            w.blockSignals(True)
            w.clear()
        for w in (self.c_class, self.c_w, self.c_h):
            w.addItem(NONE, None)
        for w in (self.c_image, self.c_class, self.c_w, self.c_h, *self.c_cols):
            for h in header:
                w.addItem(h, h)
        self.c_cols[0].setCurrentIndex(-1)

        def setc(cb, val):
            i = cb.findData(val)
            cb.setCurrentIndex(i if i >= 0 else 0)

        setc(self.c_image, m.get("image"))
        setc(self.c_class, m.get("class"))
        setc(self.c_w, m.get("width"))
        setc(self.c_h, m.get("height"))
        box, poly = m.get("box"), m.get("poly")
        geo = "none"
        cols = []
        if box:
            geo = box["format"]
            cols = box.get("cols") or [box.get("col")]
        elif poly:
            geo, cols = "poly", [poly["col"]]
        self.c_geo.blockSignals(True)
        self.c_geo.setCurrentIndex(max(0, self.c_geo.findData(geo)))
        self.c_geo.blockSignals(False)
        for i, cb in enumerate(self.c_cols):
            setc(cb, cols[i] if i < len(cols) else (header[0] if header else None))
        self.c_unit.setCurrentIndex(0)
        for w in (self.c_image, self.c_class, self.c_w, self.c_h, *self.c_cols):
            w.blockSignals(False)
        self._geo_visibility()

    def _geo_visibility(self):
        geo = self.c_geo.currentData()
        n = {"xyxy": 4, "xywh": 4, "cxcywh": 4, "str": 1, "poly": 1, "none": 0}.get(geo, 0)
        for i, cb in enumerate(self.c_cols):
            cb.setVisible(i < n)

    def _mapping(self):
        geo = self.c_geo.currentData()
        cols = [cb.currentData() for cb in self.c_cols]
        m = {"image": self.c_image.currentData(), "class": self.c_class.currentData(), "width": self.c_w.currentData(),
             "height": self.c_h.currentData(), "unit": self.c_unit.currentData() or "auto", "box": None, "poly": None}
        if geo in ("xyxy", "xywh", "cxcywh") and all(cols):
            m["box"] = {"format": geo, "cols": cols}
        elif geo == "str" and cols[0]:
            m["box"] = {"format": "str", "col": cols[0], "order": "xywh"}
        elif geo == "poly" and cols[0]:
            m["poly"] = {"col": cols[0]}
        return m

    def _mapping_changed(self, *_):
        self._geo_visibility()
        self._reparse()

    # ------------------------------------------------------------------ разбор и предпросмотр
    def _options(self) -> AI.Options:
        o = AI.Options(existing="replace" if self.r_replace.isChecked() else "skip" if self.r_skip.isChecked() else "append",
                       add_missing_images=self.k_add.isChecked(), import_tags=self.k_tags.isChecked(), images_dir=self.images_dir)
        det = self.current_det()
        if det is not None and det.parser == "table":
            o.params["mapping"] = self._mapping()
            o.params["unit"] = self.c_unit.currentData() or "auto"
        for r in range(self.cls_table.rowCount()):
            src = self.cls_table.item(r, 0).data(Qt.UserRole)
            cb = self.cls_table.cellWidget(r, 2)
            i = cb.currentIndex()
            custom = i < 0 or cb.currentText() != cb.itemText(i)        # пользователь вписал своё название
            tag = None if custom else cb.itemData(i)
            if tag == "skip":
                o.class_map[src] = None
            elif custom:
                if cb.currentText().strip() and cb.currentText().strip() != src:
                    o.class_map[src] = cb.currentText().strip()
            elif tag not in ("same", None) and tag != src:
                o.class_map[src] = tag
        return o

    def _reparse(self):
        det = self.current_det()
        if det is None:
            return
        if self._busy:
            self._pending = True
            return
        self._busy_on("Читаю разметку…")
        self._thread = _Parse(self.ctx, det, self._options(), self.project)
        self._thread.done.connect(self._parsed)
        self._thread.failed.connect(self._failed)
        self._thread.start()

    def _parsed(self, parsed, pv):
        self._busy_off()
        self.parsed, self.pv = parsed, pv
        self.status.setText("")
        lines = [f"<p><b>Записей о картинках:</b> {pv.records} · <b>объектов:</b> {pv.objects} · "
                 f"<b>картинки найдены:</b> {pv.matched} из {pv.records}"
                 + (f" · <span style='color:#e0a030'>не найдены: {pv.unmatched}</span>" if pv.unmatched else "")
                 + (f" · неоднозначных: {pv.ambiguous}" if pv.ambiguous else "") + "</p>"]
        if pv.widened:
            lines.append("<p style='color:#8b92a1'>Картинки искались и в папке выше: рядом с разметкой их не оказалось.</p>")
        if pv.unmatched_samples:
            lines.append("<p style='color:#8b92a1'>Например не найдено: " + ", ".join(html.escape(str(s)) for s in pv.unmatched_samples[:5]) +
                         ". Укажите папку с картинками кнопкой выше, если они лежат отдельно.</p>")
        for n in pv.notes[:6]:
            lines.append(f"<p style='color:#8b92a1'>{html.escape(str(n))}</p>")
        self.info.setHtml("".join(lines))
        self._fill_classes(parsed)
        self.go.setEnabled(pv.records > 0 and (pv.matched > 0 or self.k_add.isChecked() and pv.records > 0))

    def _fill_classes(self, parsed):
        keep = {}
        for r in range(self.cls_table.rowCount()):                  # сохраняем выбор пользователя при перечитывании
            cbk = self.cls_table.cellWidget(r, 2)
            ik = cbk.currentIndex()
            keep[self.cls_table.item(r, 0).data(Qt.UserRole)] = (cbk.currentText() if ik < 0 or cbk.currentText() != cbk.itemText(ik)
                                                                 else cbk.itemData(ik))
        keys = AI.class_keys(parsed)
        counts: dict = {}
        for rec in parsed.records:
            for o in rec.objs:
                if o.cls is not None:
                    k = AI.name_of(o.cls, parsed)
                    counts[k] = counts.get(k, 0) + 1
            if rec.tag is not None:
                k = AI.name_of(rec.tag, parsed)
                counts[k] = counts.get(k, 0) + 1
        self.cls_table.setRowCount(len(keys))
        existing = self.project.class_names()
        for r, k in enumerate(keys):
            it = QTableWidgetItem(k)
            it.setData(Qt.UserRole, k)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.cls_table.setItem(r, 0, it)
            c = QTableWidgetItem(str(counts.get(k, 0)))
            c.setFlags(c.flags() & ~Qt.ItemIsEditable)
            self.cls_table.setItem(r, 1, c)
            cb = QComboBox()
            cb.setEditable(True)
            cb.addItem(NEW_CLASS, "same")
            cb.addItem(SKIP_CLASS, "skip")
            for n in existing:
                cb.addItem(n, n)
            want = keep.get(k, k if k in existing else "same")
            i = cb.findData(want)
            cb.setCurrentIndex(i) if i >= 0 else cb.setEditText(str(want))
            self.cls_table.setCellWidget(r, 2, cb)

    # ------------------------------------------------------------------ запись
    def _run(self):
        det, parsed = self.current_det(), self.parsed
        if det is None or parsed is None or self._busy:
            return
        self._busy_on("Записываю в проект…")
        self.bar.setRange(0, max(1, len(parsed.records)))
        self._thread = _Apply(self.project, self.ctx, det, self._options(), parsed)
        self._thread.progress.connect(lambda i, n: (self.bar.setRange(0, max(1, n)), self.bar.setValue(i)))
        self._thread.done.connect(self._applied)
        self._thread.failed.connect(self._failed)
        self._thread.start()

    def _applied(self, rep):
        self._busy_off()
        self.report = rep
        text = rep.summary()
        if rep.unmatched:
            text += "\n\nНе найдены картинки (первые): " + ", ".join(map(str, rep.unmatched[:6]))
        QMessageBox.information(self, "Импорт разметки", text)
        self.accept()

    def reject(self):
        th = self._thread
        if th is not None and th.isRunning():
            if isinstance(th, _Apply):
                th.cancelled = True                                  # запись останавливается между картинками
            QApplication.setOverrideCursor(Qt.WaitCursor)
            th.wait(8000)
            QApplication.restoreOverrideCursor()
            if th.isRunning():
                return                                               # ещё работает: окно не закрываем, чтобы не оборвать запись
        super().reject()
