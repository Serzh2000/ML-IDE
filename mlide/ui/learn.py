"""Справочник, советник по настройкам, запуск и «живое» объяснение обучения."""
from __future__ import annotations

from . import winflags

import csv
import html
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import paths, textio
from .. import knowledge as K
from ..i18n import T
from . import icons, style

def find_python() -> str:
    """Default interpreter for training. A path under WindowsApps is usually the Microsoft Store stub
    (it opens the Store instead of running anything), so any other interpreter is preferred over it."""
    fallback = ""
    for name in ("python", "python3", "py"):
        p = shutil.which(name)
        if p and "windowsapps" not in p.lower():
            return p
        fallback = fallback or p or ""
    return fallback


PROC_ERRORS = {
    QProcess.FailedToStart: "не удалось запустить - проверьте путь к Python (файл не найден или это не программа)",
    QProcess.Crashed: "процесс аварийно завершился",
}


COLOR = {"ok": "#3fb950", "warn": "#f0a030", "info": "#58a6ff"}


def cards_html(items):
    """items: (level, title, text) -> html с цветными карточками."""
    out = []
    for lvl, title, text in items:
        out.append(f"<table width='100%' cellpadding='6' style='margin-bottom:6px'><tr>"
                   f"<td width='4' bgcolor='{COLOR[lvl]}'></td><td>"
                   f"<b>{icons.status_img(lvl)}&nbsp;{title}</b><br>{text}</td></tr></table>")
    return "".join(out)


# =============================================================================== справочник
class HelpDialog(QDialog):
    def __init__(self, shortcuts_html: str, parent=None, start: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Справочник")
        self.resize(980, 640)
        self.entries: list[tuple[str, str, str]] = []     # (section, title, html)
        for t, h in K.GUIDES:
            self.entries.append(("Руководства", t, f"<h2>{t}</h2>{h}"))
        self.entries.append(("Руководства", "Горячие клавиши", shortcuts_html))
        for g in K.GLOSSARY:
            self.entries.append(("Словарь терминов", g[0], K.glossary_html(g)))

        lay = QHBoxLayout(self)
        sp = QSplitter()
        lay.addWidget(sp)
        left = QWidget()
        v = QVBoxLayout(left)
        v.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск: mAP, переобучение, batch…")
        self.search.textChanged.connect(self._fill)
        self.lst = QListWidget()
        v.addWidget(self.search)
        v.addWidget(self.lst, 1)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        sp.addWidget(left)
        sp.addWidget(self.view)
        sp.setSizes([300, 680])
        self.lst.currentItemChanged.connect(self._show)
        self._fill()
        if start:
            for r in range(self.lst.count()):
                if self.lst.item(r).text().startswith(start):
                    self.lst.setCurrentRow(r)
                    break

    def _fill(self):
        q = self.search.text().strip().lower()
        self.lst.blockSignals(True)
        self.lst.clear()
        last = None
        for sec, title, h in self.entries:
            if q and q not in title.lower() and q not in re.sub("<[^>]+>", " ", h).lower():
                continue
            if sec != last:
                it = QListWidgetItem(T(sec).upper())
                it.setFlags(Qt.NoItemFlags)
                f = it.font()
                f.setBold(True)
                f.setPointSize(8)
                it.setFont(f)
                self.lst.addItem(it)
                last = sec
            it = QListWidgetItem("   " + title)
            it.setData(Qt.UserRole, h)
            self.lst.addItem(it)
        self.lst.blockSignals(False)
        for r in range(self.lst.count()):
            if self.lst.item(r).data(Qt.UserRole):
                self.lst.setCurrentRow(r)
                break
        else:
            self.view.setHtml("<p>Ничего не найдено.</p>")

    def _show(self, cur, _):
        if cur is not None and cur.data(Qt.UserRole):
            self.view.setHtml(cur.data(Qt.UserRole))


# =============================================================================== советник
class AdvisorDialog(QDialog):
    def __init__(self, project, parent=None, can_apply=True):
        super().__init__(parent)
        self.setWindowTitle("Советник по настройкам")
        self.resize(720, 640)
        self.settings, advice = K.analyze(project)
        lay = QVBoxLayout(self)
        intro = QLabel("Анализ вашего датасета: что в нём хорошо, что стоит исправить и какие настройки обучения "
                       "подойдут. Это эвристики на основе практики - ориентир, а не закон.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        v = QTextBrowser()
        v.setHtml(cards_html([(a.level, a.title, a.text) for a in advice]))
        lay.addWidget(v, 1)
        b = QDialogButtonBox()
        if self.settings and can_apply:
            b.addButton("Применить к экспорту", QDialogButtonBox.AcceptRole)
        b.addButton(QDialogButtonBox.Close)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        lay.addWidget(b)


# =============================================================================== графики
def fill_quality_chart(chart, S):
    """Детекция: mAP50, precision, recall. Классификация (нет mAP, есть accuracy): точность top-1."""
    if S.get("acc") and not any(v for v in S["map50"] if v is not None):
        chart.series = [("acc", "точность top-1 (accuracy)", "#2ecc71")]
        chart.set_data(S["epoch"], {"acc": S["acc"]})
    else:
        chart.series = [("map50", "mAP50", "#2ecc71"), ("p", "precision", "#a29bfe"), ("r", "recall", "#fd79a8")]
        chart.set_data(S["epoch"], {"map50": S["map50"], "p": S["p"], "r": S["r"]})


class LineChart(QWidget):
    def __init__(self, title, hint, series, parent=None):
        super().__init__(parent)
        self.title, self.hint, self.series = title, hint, series     # series: [(key, label, color)]
        self.xs, self.data = [], {}
        self.setMinimumHeight(170)
        self.setToolTip(hint)

    def set_data(self, xs, data):
        self.xs, self.data = xs, data
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(style.CHART))
        p.setPen(QColor(style.TEXT))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(10, 16, self.title)
        # легенда
        x = 10
        p.setFont(QFont("Segoe UI", 8))
        for key, label, col in self.series:
            p.setPen(QPen(QColor(style.chart_color(col)), 3))
            p.drawLine(x, 30, x + 14, 30)
            p.setPen(QColor(style.MUTED))
            p.drawText(x + 18, 34, label)
            x += 24 + p.fontMetrics().horizontalAdvance(T(label)) + 8
        r = QRectF(44, 42, self.width() - 56, self.height() - 62)
        vals = [v for k, _l, _c in self.series for v in self.data.get(k, []) if v is not None and np.isfinite(v)]
        if not vals or len(self.xs) < 1:
            p.setPen(QColor(style.MUTED))
            p.drawText(r, Qt.AlignCenter, "ждём данные первой эпохи…")
            return
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1
        pad = (hi - lo) * 0.08
        lo, hi = max(0, lo - pad) if lo >= 0 else lo - pad, hi + pad
        x0, x1 = min(self.xs), max(self.xs)
        if x1 == x0:
            x1 = x0 + 1
        p.setPen(QPen(QColor(style.BORDER), 1))
        for i in range(5):
            y = r.top() + r.height() * i / 4
            p.drawLine(r.left(), y, r.right(), y)
            p.setPen(QColor(style.MUTED))
            p.drawText(2, int(y + 4), f"{hi - (hi - lo) * i / 4:.2f}")
            p.setPen(QPen(QColor(style.BORDER), 1))
        p.setPen(QColor(style.MUTED))
        p.drawText(int(r.left()), int(r.bottom() + 14), f"эпоха {x0:.0f}")
        p.drawText(int(r.right() - 60), int(r.bottom() + 14), f"эпоха {x1:.0f}")
        for key, _l, col in self.series:
            ys = self.data.get(key, [])
            pts = [QPointF(r.left() + (xv - x0) / (x1 - x0) * r.width(),
                           r.bottom() - (yv - lo) / (hi - lo) * r.height())
                   for xv, yv in zip(self.xs, ys) if yv is not None and np.isfinite(yv)]
            if len(pts) >= 2:
                p.setPen(QPen(QColor(style.chart_color(col)), 2))
                p.drawPolyline(QPolygonF(pts))
            elif pts:
                p.setBrush(QColor(style.chart_color(col)))
                p.drawEllipse(pts[0], 3, 3)


# =============================================================================== обучение


def diagnose_exit(code, log):
    """Переводит код выхода и хвост журнала в понятное объяснение: (заголовок, html-текст)."""
    c = code & 0xFFFFFFFF
    low = log.lower()
    why, todo = [], []
    if "'val:' key missing" in log or "'train' and 'val' are required" in low:
        why.append("В data.yaml нет проверочной части (val).")
        todo.append("Сделайте экспорт заново (окно «Экспорт датасета»): новая версия всегда записывает val. "
                    "Если картинок мало или все похожи - добавьте ещё изображений.")
    if "no module named" in low and "ultralytics" in low:
        why.append("Не установлен пакет ultralytics.")
        todo.append("Нажмите «Установить ultralytics».")
    if "out of memory" in low or "winerror 1455" in low or "paging file" in low:
        why.append("Не хватило памяти (видеопамяти или оперативной).")
        todo.append("В экспорте уменьшите batch (например до 4) и размер изображения (например до 416), закройте другие программы.")
    if "cuda" in low and ("error" in low or "not available" in low) and not why:
        why.append("Проблема с видеокартой или драйвером CUDA.")
        todo.append("Нажмите «Видеокарта…» и установите PyTorch под вашу видеокарту, либо обучайте на процессоре.")
    if c == 0xC0000005 and not why:
        why.append("Python аварийно завершился (ошибка доступа к памяти, код 0xC0000005). Обычно так бывает, "
                   "когда не хватает оперативной памяти, при сбое установленного PyTorch или из-за антивируса.")
        todo.append("В экспорте поставьте batch 4 и размер изображения 416, закройте другие программы и запустите снова. "
                    "Если повторяется - нажмите «Установить ultralytics» (переустановит пакеты) или запустите train.py "
                    "в окне командной строки и пришлите последние строки журнала.")
    elif c in (0xC000013A, 0xC0000142) and not why:
        why.append("Процесс был закрыт (окно, антивирус или завершение Python).")
    if not why:
        why.append(f"Обучение завершилось с кодом {code}.")
        todo.append("Последние строки журнала внизу показывают причину.")
    err = ""
    for ln in reversed([x.strip() for x in log.splitlines() if x.strip()]):
        if re.search(r"(error|exception|ошибка)", ln, re.I):
            err = ln[:300]
            break
    text = " ".join(html.escape(x) for x in why) + "<br><br><b>Что делать:</b> " + " ".join(html.escape(x) for x in todo)
    if err:
        text += "<br><br><i>Последняя ошибка в журнале:</i> " + html.escape(err)
    return "Обучение прервалось", text


class TrainDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Обучение модели")
        scr = self.screen().availableGeometry() if self.screen() else None
        self.resize(min(1180, int(scr.width() * 0.95)) if scr else 1180, min(780, int(scr.height() * 0.9)) if scr else 780)
        self.project = project
        self.proc: QProcess | None = None
        self.total_epochs = 0
        self.started_mtime = 0.0
        self._last_csv = None

        lay = winflags.scroll_layout(self)
        intro = QLabel("Запускает <code>train.py</code> из папки экспорта (формат Ultralytics), рисует графики и "
                       "<b>объясняет простым языком</b>, что сейчас происходит и что делать. Нужны Python с пакетом "
                       "<code>ultralytics</code> (кнопка «Установить» сделает это) - желательно с видеокартой Nvidia.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        row = QHBoxLayout()
        self._synced_dir = project.export.get("out_dir", "")
        self.folder = QLineEdit(self._synced_dir)
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(b)
        f.addRow("Папка экспорта (с train.py)", row)
        row = QHBoxLayout()
        self.py = QLineEdit(find_python())
        b = QPushButton("Проверить")
        b.clicked.connect(self.check_env)
        b2 = QPushButton("Установить ultralytics")
        b2.clicked.connect(self.install)
        row.addWidget(self.py, 1)
        bg = QPushButton("Видеокарта…")
        bg.setToolTip("<p>Есть ли у вас видеокарта Nvidia и как включить её для обучения (CUDA); бесплатные варианты с GPU.</p>")
        bg.clicked.connect(lambda: __import__("mlide.ui.help_dialogs", fromlist=["x"]).GpuHelpDialog(self).exec())
        row.addWidget(b)
        row.addWidget(b2)
        row.addWidget(bg)
        f.addRow("Python", row)
        lay.addLayout(f)

        row = QHBoxLayout()
        self.start_btn = QPushButton("Начать обучение")
        self.start_btn.setIcon(icons.icon("start"))
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setIcon(icons.icon("stop"))
        self.stop_btn.clicked.connect(self._stop_clicked)
        self.stop_btn.setEnabled(False)
        ob = QPushButton("Открыть папку результатов")
        ob.clicked.connect(self.open_results)
        self.status = QLabel("")
        for _b in (self.start_btn, self.stop_btn, ob):
            _b.setAutoDefault(False)
            _b.setMinimumWidth(_b.sizeHint().width() + 12)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(ob)
        row.addWidget(self.status, 1)
        lay.addLayout(row)

        sp = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.c_loss = LineChart("Ошибка (loss) - чем меньше, тем лучше",
                                "Ошибка модели. Должна плавно снижаться. Если на проверке (val) растёт - переобучение.",
                                [("tbox", "обучение (train)", "#4da3ff"), ("vbox", "проверка (val)", "#ff9f43")])
        self.c_map = LineChart("Качество - чем больше, тем лучше",
                               "mAP50 - общее качество; precision - доля верных находок; recall - доля найденных объектов.",
                               [("map50", "mAP50", "#2ecc71"), ("p", "precision", "#a29bfe"), ("r", "recall", "#fd79a8")])
        lv.addWidget(self.c_loss)
        lv.addWidget(self.c_map)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        lv.addWidget(QLabel("Журнал"))
        lv.addWidget(self.log, 1)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("<b>Что сейчас происходит</b>"))
        self.explain = QTextBrowser()
        rv.addWidget(self.explain, 1)
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setSizes([640, 520])
        lay.addWidget(sp, 1)

        self.folder.setToolTip("<p>Папка, в которую вы экспортировали датасет: в ней лежат train.py, data.yaml и появится runs/.</p>")
        self.py.setToolTip("<p>Путь к python.exe, в котором установлен ultralytics. Если пакета нет - нажмите «Установить ultralytics».</p>")
        self.start_btn.setToolTip("<p>Запустить train.py. Окно можно закрыть: обучение продолжится в фоне.</p>")
        self.stop_btn.setToolTip("<p>Остановить обучение. Лучшая версия модели к этому моменту уже сохранена в weights/best.pt.</p>")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(2000)
        self._first_load()

    # ---- helpers
    def set_project(self, project):
        """Called every time the dialog is shown: follow the project's latest export folder (unless busy)."""
        self.project = project
        d = project.export.get("out_dir", "")
        if not self.running and d and d != self._synced_dir:
            self._synced_dir = d
            self.folder.setText(d)
            self._first_load()

    def _dir(self) -> Path:
        return Path(self.folder.text().strip())

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Папка экспорта", self.folder.text())
        if d:
            self.folder.setText(d)
            self._first_load()

    def _append(self, text):
        self.log.appendPlainText(text)

    def _find_csv(self):
        d = self._dir() / "runs"
        if not d.exists():
            return None
        files = sorted(d.rglob("results.csv"), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def _read_total_epochs(self):
        try:
            t = (self._dir() / "train.py").read_text(encoding="utf-8")
            m = re.search(r"epochs=(\d+)", t)
            self.total_epochs = int(m.group(1)) if m else 0
        except Exception:
            self.total_epochs = 0

    @property
    def running(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def _first_load(self):
        self._read_total_epochs()
        self.poll(force=True)

    # ---- environment
    def _run_quick(self, args, title):
        py = self.py.text().strip()
        if not py:
            QMessageBox.warning(self, title, "Укажите путь к Python (python.exe).")
            return
        pr = QProcess(self)
        env = paths.child_qenv()
        env.insert("PYTHONIOENCODING", "utf-8")
        pr.setProcessEnvironment(env)
        pr.setProcessChannelMode(QProcess.MergedChannels)
        pr.finished.connect(lambda *_: (self._append(textio.decode_stream(pr, pr.readAllStandardOutput())),
                                        self.status.setText("")))

        def failed(err):
            if err == QProcess.FailedToStart:       # `finished` is never emitted in this case
                self._append(f"--- {title}: {PROC_ERRORS[err]}: {py} ---")
                self.status.setText("")
        pr.errorOccurred.connect(failed)
        self.status.setText(f"{title}…")
        with paths.child_launch():           # без каталога DLL приложения (иначе Python падает с 0xC0000005)
            pr.start(py, args)
            pr.waitForStarted(8000)
        return pr

    def check_env(self):
        self._append("--- проверка окружения ---")
        self._chk = self._run_quick(["-c", "import sys;print(sys.version);import ultralytics,torch;"
                                           "print('ultralytics',ultralytics.__version__,'torch',torch.__version__,"
                                           "'CUDA доступна:',torch.cuda.is_available())"], "Проверка")

    def install(self):
        if QMessageBox.question(self, "Установка", "Будет выполнено <b>pip install ultralytics</b> (скачает несколько "
                                "сотен МБ, включая PyTorch). Продолжить?") != QMessageBox.Yes:
            return
        self._append("--- установка ultralytics (это может занять несколько минут) ---")
        self._inst = self._run_quick(["-m", "pip", "install", "ultralytics"], "Установка")
        if self._inst:
            self._inst.readyReadStandardOutput.connect(
                lambda: self._append(textio.decode_stream(self._inst, self._inst.readAllStandardOutput()).rstrip()))

    # ---- start / stop
    def start(self):
        if self.running:
            return
        if not self.folder.text().strip():
            QMessageBox.warning(self, "Обучение", "Укажите папку экспорта (в ней лежит train.py).")
            return
        d = self._dir()
        if not (d / "train.py").exists():
            box = QMessageBox(QMessageBox.Question, "Обучение", "В папке нет train.py. Нужно сделать экспорт в формате "
                              "Ultralytics с галочкой «Создать train.py». Открыть окно экспорта сейчас?", parent=self)
            go = box.addButton("Открыть экспорт", QMessageBox.AcceptRole)
            box.addButton("Отмена", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is not go:
                return
            main = self.parent()
            if main is None or not hasattr(main, "export"):
                return
            main.export()                                   # то же окно, что Ctrl+E
            out = self.project.export.get("out_dir", "")
            if out:
                self.folder.setText(out)
                self._synced_dir = out
            if (self._dir() / "train.py").exists():
                self.raise_()
                self.start()                                # экспорт готов - сразу продолжаем обучение
            return
        py = self.py.text().strip()
        if not py:
            QMessageBox.warning(self, "Обучение", "Укажите путь к Python.")
            return
        self._read_total_epochs()
        if not getattr(self, "_gpu_checked", False) and (self.total_epochs == 0 or self.total_epochs > 10):
            from .help_dialogs import GpuHelpDialog, detect_gpu
            if not detect_gpu().get("found"):
                box = QMessageBox(QMessageBox.Warning, "Обучение",
                                  "Видеокарта Nvidia не найдена, обучение пойдёт на процессоре. Это в десятки раз медленнее: полное обучение "
                                  "(100 эпох) может занять много часов. Для пробы в экспорте поставьте 5-10 эпох и меньший размер изображения.",
                                  parent=self)
                go = box.addButton("Всё равно начать", QMessageBox.AcceptRole)
                gpu = box.addButton("Про видеокарту…", QMessageBox.ActionRole)
                box.addButton("Отмена", QMessageBox.RejectRole)
                box.exec()
                if box.clickedButton() is gpu:
                    GpuHelpDialog(self).exec()
                    return
                if box.clickedButton() is not go:
                    return
            self._gpu_checked = True                        # предупреждаем один раз за сеанс окна
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(d))
        env = paths.child_qenv()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_out)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        import time
        self.started_mtime = time.time()
        self.log.clear()
        self.c_loss.set_data([], {})
        self.c_map.set_data([], {})
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("идёт обучение…")
        self.explain.setHtml(cards_html(K.explain_training(K.parse_rows([]), self.total_epochs)))
        # last: a failed start reports through errorOccurred synchronously, from inside start()
        with paths.child_launch():           # без каталога DLL приложения (иначе Python падает с 0xC0000005)
            self.proc.start(py, ["-u", "train.py"])
            self.proc.waitForStarted(8000)

    def _stop_clicked(self):
        if not self.running:
            return
        if QMessageBox.question(self, "Остановить обучение?", "Обучение будет прервано. Уже сохранённые эпохи останутся в папке runs "
                                "(weights/last.pt и best.pt), продолжить с того же места нельзя без повторного запуска. Остановить?") == QMessageBox.Yes:
            self.stop()

    def stop(self, wait=False):
        if not self.running:
            return
        pid = self.proc.processId()
        if sys.platform == "win32" and pid:
            # kill() ends only the main python.exe; DataLoader workers would live on and keep the GPU busy
            import subprocess
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=10,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception:
                pass
        self.proc.kill()
        self._append("--- остановлено пользователем (лучшая версия уже сохранена в weights/best.pt) ---")
        if wait:
            self.proc.waitForFinished(5000)

    def _on_error(self, err):
        if err != QProcess.FailedToStart:           # other errors are followed by `finished`
            return
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("не запущено")
        msg = f"{PROC_ERRORS[err]}: {self.py.text().strip()}"
        self._append(f"--- {msg} ---")
        self.explain.setHtml(cards_html([("warn", "Обучение не запустилось", html.escape(msg))]))

    def _on_out(self):
        text = textio.decode_stream(self.proc, self.proc.readAllStandardOutput())
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        for ln in re.split(r"[\r\n]+", text):
            if ln.strip():
                self._append(ln.rstrip())

    def _on_finished(self, code, _status):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("завершено" if code == 0 else "обучение прервано, причина - справа")
        if code != 0:
            title, text = diagnose_exit(code, self.log.toPlainText()[-6000:])
            self.explain.setHtml(cards_html([("warn", title, text)]))
        self.poll(force=True, finished=(code == 0))

    # ---- polling results.csv
    def poll(self, force=False, finished=False):
        if not (self.running or force):
            return
        csvp = self._find_csv()
        if not csvp:
            return
        if self.running and csvp.stat().st_mtime < self.started_mtime - 5:
            return                                       # старый результат, ждём новый
        try:
            with open(csvp, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return
        S = K.parse_rows(rows)
        self.c_loss.set_data(S["epoch"], {"tbox": S["tbox"], "vbox": S["vbox"]})
        fill_quality_chart(self.c_map, S)
        self.explain.setHtml(cards_html(K.explain_training(S, self.total_epochs, finished=finished)))

    def open_results(self):
        d = self._dir() / "runs"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d if d.exists() else self._dir())))

    def closeEvent(self, e):
        if self.running:
            self.hide()            # обучение продолжается в фоне; окно можно открыть снова из меню
            e.ignore()
        else:
            super().closeEvent(e)

    def reject(self):
        self.close()
