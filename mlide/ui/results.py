"""Окно «Результаты обучения»: метрики, графики, картинки Ultralytics, проверка модели на фото."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QScrollArea, QSplitter, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from .. import autolabel, imgio
from .. import knowledge as K
from ..model import PALETTE
from .dialogs import bgr_to_qpixmap, hex_to_bgr
from .learn import LineChart, cards_html, fill_quality_chart

EXPLAIN_IMG = {
    "results.png": ("Все графики обучения", "Кривые ошибок (loss) по эпохам и метрик качества. Loss должны снижаться, метрики - расти и выходить на плато."),
    "confusion_matrix_normalized.png": ("Матрица ошибок (в долях)", "Столбцы - что на самом деле, строки - что предсказала модель. По диагонали должны быть значения около 1.0. Пятна вне диагонали - какие классы путаются; строка/столбец background - пропуски и ложные срабатывания."),
    "confusion_matrix.png": ("Матрица ошибок (в штуках)", "То же, но в абсолютных числах: сколько объектов угадано и сколько перепутано."),
    "BoxPR_curve.png": ("Кривая Precision-Recall", "Чем ближе кривая к правому верхнему углу, тем лучше. Площадь под ней и есть mAP50. Если у одного класса кривая заметно ниже остальных - ему не хватает данных."),
    "BoxF1_curve.png": ("F1 по порогу уверенности", "Максимум кривой - лучший порог confidence для работы модели (обычно 0.3-0.5)."),
    "BoxP_curve.png": ("Precision по порогу уверенности", "Как растёт точность при повышении порога confidence."),
    "BoxR_curve.png": ("Recall по порогу уверенности", "Как падает полнота при повышении порога confidence."),
    "labels.jpg": ("Статистика вашей разметки", "Число объектов по классам, положение и размеры рамок. Сильный перекос в числе объектов - дисбаланс классов."),
    "val_batch0_labels.jpg": ("Проверка: ваша разметка", "Проверочные картинки с вашей разметкой. Сравните с «Проверка: ответы модели»."),
    "val_batch0_pred.jpg": ("Проверка: ответы модели", "Те же картинки, но с рамками, которые нарисовала модель. Число рядом - её уверенность."),
    "train_batch0.jpg": ("Примеры обучения (после аугментации)", "Так модель видит обучающие картинки: мозаика, повороты, сдвиги, изменение цвета."),
}
_ORDER = list(EXPLAIN_IMG)


class ResultsDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Результаты обучения")
        self.resize(1300, 840)
        self.project = project
        self.run_dir: Path | None = None
        self._pix = None
        self._test_pix = None

        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.folder = QLineEdit(project.export.get("out_dir", ""))
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        self.runs = QComboBox()
        self.runs.setMinimumWidth(220)
        self.runs.currentIndexChanged.connect(self._load_run)
        row.addWidget(QLabel("Папка экспорта / обучения:"))
        row.addWidget(self.folder, 1)
        row.addWidget(b)
        row.addWidget(QLabel("Запуск:"))
        row.addWidget(self.runs)
        eb = QPushButton("Экспортировать модель…")
        eb.setToolTip("<p>Выгрузить лучшую модель этого запуска в ONNX, OpenVINO, TensorRT, NCNN, TFLite, ESP32 и др.</p>")
        eb.clicked.connect(self._export_model)
        row.addWidget(eb)
        vb = QPushButton("Схема модели…")
        vb.setToolTip("<p>Показать устройство обученной модели: слои, связи, число параметров и пояснения. Нужен best.onnx.</p>")
        vb.clicked.connect(self._view_model)
        row.addWidget(vb)
        cb = QPushButton("Сравнить запуски…")
        cb.setToolTip("<p>Таблица и наложенные кривые всех запусков обучения из папки: какой лучше и чем отличались настройки.</p>")
        cb.clicked.connect(self._compare)
        row.addWidget(cb)
        lb = QPushButton("Видео / камера…")
        lb.setToolTip("<p>Прогнать лучшую модель по видеофайлу или веб-камере и замерить скорость (кадр/с).</p>")
        lb.clicked.connect(self._live)
        row.addWidget(lb)
        lay.addLayout(row)
        self.summary = QLabel("")
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)

        sp = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.c_loss = LineChart("Ошибка (loss) - чем меньше, тем лучше", "",
                                [("tbox", "обучение (train)", "#4da3ff"), ("vbox", "проверка (val)", "#ff9f43")])
        self.c_map = LineChart("Качество - чем больше, тем лучше", "",
                               [("map50", "mAP50", "#2ecc71"), ("p", "precision", "#a29bfe"), ("r", "recall", "#fd79a8")])
        lv.addWidget(self.c_loss)
        lv.addWidget(self.c_map)
        lv.addWidget(QLabel("<b>Разбор результатов</b>"))
        self.explain = QTextBrowser()
        lv.addWidget(self.explain, 1)
        sp.addWidget(left)

        self.tabs = QTabWidget()
        t1 = QWidget()
        v1 = QVBoxLayout(t1)
        self.files = QComboBox()
        self.files.currentIndexChanged.connect(self._show_file)
        self.img_note = QLabel("")
        self.img_note.setWordWrap(True)
        self.img = QLabel("")
        self.img.setAlignment(Qt.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.img)
        v1.addWidget(self.files)
        v1.addWidget(self.img_note)
        v1.addWidget(self.scroll, 1)
        self.tabs.addTab(t1, "Графики и примеры")

        t2 = QWidget()
        v2 = QVBoxLayout(t2)
        r2 = QHBoxLayout()
        pb = QPushButton("Проверить модель на фото…")
        pb.clicked.connect(self._test_model)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        pb.setToolTip("<p>Выберите любое фото: лучшая модель этого запуска найдёт на нём объекты.</p>")
        r2.addWidget(pb)
        self.conf.setToolTip("<p>Показывать находки, в которых модель уверена не меньше этого значения. Если ничего не найдено - снизьте.</p>")
        r2.addWidget(QLabel("Порог уверенности:"))
        r2.addWidget(self.conf)
        r2.addStretch(1)
        v2.addLayout(r2)
        self.test_info = QLabel("Выберите любую фотографию: лучшая модель этого запуска (best.onnx / best.pt) найдёт "
                                "на ней объекты. Так видно, как она поведёт себя на новых картинках.")
        self.test_info.setWordWrap(True)
        self.test_img = QLabel("")
        self.test_img.setAlignment(Qt.AlignCenter)
        self.test_img.setMinimumSize(300, 300)
        v2.addWidget(self.test_info)
        v2.addWidget(self.test_img, 1)
        self.tabs.addTab(t2, "Проверка модели")
        sp.addWidget(self.tabs)
        sp.setSizes([560, 700])
        lay.addWidget(sp, 1)
        self._scan()

    # ---------------------------------------------------------------- поиск запусков
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Папка экспорта или обучения", self.folder.text())
        if d:
            self.folder.setText(d)
            self._scan()

    def _scan(self):
        base = Path(self.folder.text().strip() or ".")
        self.runs.blockSignals(True)
        self.runs.clear()
        found = []
        if base.exists():
            found = sorted(base.rglob("results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in found:
            try:
                label = str(p.parent.relative_to(base))
            except ValueError:
                label = str(p.parent)
            self.runs.addItem(label if label != "." else p.parent.name, str(p.parent))
        self.runs.blockSignals(False)
        if found:
            self._load_run()
            return
        self.run_dir = None
        self.summary.setText("<span style='color:#f0a030'>Результаты не найдены.</span> Укажите папку экспорта, "
                             "в которой есть <code>runs/…/results.csv</code> (он появляется после обучения).")
        for c in (self.c_loss, self.c_map):
            c.set_data([], {})
        self.explain.setHtml("")
        self.files.clear()
        self.img.clear()
        self.img_note.setText("")

    def _load_run(self):
        d = self.runs.currentData()
        if not d:
            return
        self.run_dir = Path(d)
        try:
            with open(self.run_dir / "results.csv", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:  # noqa
            self.summary.setText(f"Не удалось прочитать results.csv: {e}")
            return
        S = K.parse_rows(rows)
        total = 0
        try:
            m = re.search(r"^epochs:\s*(\d+)", (self.run_dir / "args.yaml").read_text(encoding="utf-8"), re.M)
            total = int(m.group(1)) if m else 0
        except Exception:
            pass
        self.c_loss.set_data(S["epoch"], {"tbox": S["tbox"], "vbox": S["vbox"]})
        fill_quality_chart(self.c_map, S)
        self.explain.setHtml(cards_html(K.explain_training(S, total or len(S["epoch"]), finished=True)))
        self.summary.setText(self._summary_html(S, total))
        self._fill_files()

    @staticmethod
    def _f(v):
        return "-" if v is None else f"{v:.3f}"

    def _summary_html(self, S, total):
        n = len(S["epoch"])
        if not n:
            return "Файл результатов пуст."
        fit = [(0.1 * (S["map50"][i] or 0) + 0.9 * (S["map"][i] or 0), i) for i in range(n)]
        bi = max(fit)[1]            # так Ultralytics выбирает эпоху для best.pt

        def row(name, i):
            return (f"<tr><td>{name}</td><td align='right'>{int(S['epoch'][i])}</td>"
                    f"<td align='right'><b>{self._f(S['map50'][i])}</b></td><td align='right'>{self._f(S['map'][i])}</td>"
                    f"<td align='right'>{self._f(S['p'][i])}</td><td align='right'>{self._f(S['r'][i])}</td></tr>")

        stopped = " (остановлено раньше срока)" if total and n < total else ""
        return (f"<b>Эпох пройдено: {n}{' из ' + str(total) if total else ''}</b>{stopped}. "
                "Лучшая модель (<code>best.pt</code>) - та, у которой выше сочетание mAP50 и mAP50-95."
                "<table cellpadding='3'><tr><th align='left'>Версия</th><th>эпоха</th><th>mAP50</th>"
                "<th>mAP50-95</th><th>precision</th><th>recall</th></tr>"
                + row("Лучшая (best.pt)", bi) + row("Последняя (last.pt)", n - 1) + "</table>")

    # ---------------------------------------------------------------- картинки
    def _fill_files(self):
        self.files.blockSignals(True)
        self.files.clear()
        names = {p.name: p for p in list(self.run_dir.glob("*.png")) + list(self.run_dir.glob("*.jpg"))}
        ordered = [n for n in _ORDER if n in names] + sorted(n for n in names if n not in EXPLAIN_IMG)
        for n in ordered:
            self.files.addItem(EXPLAIN_IMG.get(n, (n, ""))[0], str(names[n]))
        self.files.blockSignals(False)
        self._show_file()

    def _show_file(self):
        p = self.files.currentData()
        if not p:
            self._pix = None
            self.img.clear()
            self.img_note.setText("В папке запуска нет картинок (их создаёт Ultralytics в конце обучения).")
            return
        self._pix = QPixmap(p)
        self.img_note.setText(EXPLAIN_IMG.get(Path(p).name, ("", ""))[1])
        self._fit_pix()

    def _fit_pix(self):
        if self._pix is not None and not self._pix.isNull():
            w = max(200, self.scroll.viewport().width() - 6)
            self.img.setPixmap(self._pix.scaledToWidth(w, Qt.SmoothTransformation))

    def _fit_test(self):
        if self._test_pix is not None:
            self.test_img.setPixmap(self._test_pix.scaled(self.test_img.size(), Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_pix()
        self._fit_test()

    def _compare(self):
        from .compare_dialog import CompareDialog
        self.project.export["out_dir"] = self.folder.text().strip() or self.project.export.get("out_dir", "")
        CompareDialog(self.project, self).exec()

    def _live(self):
        from .live_test import LiveTestDialog
        w = self._find_model()
        if w:
            self.project.export["auto_model"] = str(w)
        LiveTestDialog(self.project, self).exec()

    def _view_model(self):
        from .model_viewer import ModelViewerDialog
        w = (self.run_dir / "weights" / "best.onnx") if self.run_dir else None
        ModelViewerDialog(self, str(w) if w and w.exists() else "").exec()

    def _export_model(self):
        from .model_export import ModelExportDialog
        w = (self.run_dir / "weights" / "best.pt") if self.run_dir else None
        ModelExportDialog(self.project, self, str(w) if w and w.exists() else "").exec()

    # ---------------------------------------------------------------- проверка модели
    def _find_model(self):
        w = (self.run_dir or Path(".")) / "weights"
        for n in ("best.onnx", "best.pt"):
            if (w / n).exists():
                return w / n
        return None

    def _class_names(self):
        base = Path(self.folder.text().strip() or ".")
        run = self.run_dir or base
        for p in (base / "data.yaml", run.parent.parent / "data.yaml", run.parent / "data.yaml"):
            try:
                n = yaml.safe_load(p.read_text(encoding="utf-8")).get("names")
                return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
            except Exception:
                continue
        return self.project.class_names()

    def _test_model(self):
        m = self._find_model()
        if not m:
            QMessageBox.information(self, "Проверка модели", "В этом запуске нет weights/best.onnx или best.pt.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "Фото для проверки", "", "Изображения (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not p:
            return
        try:
            model = autolabel.load_model(m)
            img = imgio.imread(p)
            dets = model.predict(img, self.conf.value(), 0.45)
        except Exception as e:  # noqa
            QMessageBox.warning(self, "Проверка модели", f"{type(e).__name__}: {e}")
            return
        names = self._class_names()
        palette = [hex_to_bgr(c) for c in PALETTE]
        counts, out = {}, img.copy()
        th = max(2, round(max(out.shape[:2]) / 400))
        for d in dets:
            col = palette[d.cls % len(palette)]
            nm = names[d.cls] if d.cls < len(names) else (d.name or str(d.cls))
            counts[nm] = counts.get(nm, 0) + 1
            x1, y1, x2, y2 = map(int, d.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
            cv2.putText(out, f"{nm} {d.conf:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, th / 3), col, max(1, th - 1), cv2.LINE_AA)
        self._test_pix = bgr_to_qpixmap(out)
        self._fit_test()
        summ = ", ".join(f"{k} ×{v}" for k, v in counts.items()) or "ничего не найдено"
        tip = "" if dets else " Попробуйте снизить порог уверенности."
        self.test_info.setText(f"Модель {m.name}: найдено - {summ}.{tip}")
