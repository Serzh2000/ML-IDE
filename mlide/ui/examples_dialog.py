"""Окно «Примеры»: каталог учебных датасетов (поиск, фильтры), скачивание без регистрации и открытие как проекта."""
from __future__ import annotations

import html
import re

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QProgressBar, QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from .. import catalog as CT
from .. import i18n, paths
from . import icons

DOMAINS = {"everyday": "Повседневное", "nature": "Природа и сельское хозяйство", "industrial": "Промышленность и техника",
           "document": "Документы и интерфейсы", "micro": "Микроскопия", "medical": "Медицина (учебные)", "game": "Видеоигры",
           "aerial": "Аэросъёмка", "thermal": "Тепловизор и рентген", "underwater": "Под водой"}
SOURCES = {"ultralytics": "Ultralytics (GitHub)", "huggingface": "Hugging Face"}


class _Worker(QThread):
    progress = Signal(str, int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, entry):
        super().__init__()
        self.entry, self._cancel = entry, False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            p = CT.build_project(self.entry, lambda s, d, t: (self.progress.emit(s, d, t), not self._cancel)[1])
            self.done.emit(str(p))
        except Exception as e:  # noqa
            self.failed.emit(str(e))


class _Refresh(QThread):
    ok = Signal(int)
    failed = Signal(str)

    def run(self):
        try:
            self.ok.emit(CT.refresh_remote())
        except Exception as e:  # noqa
            self.failed.emit(str(e))


class ExamplesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Примеры")
        self.resize(1040, 680)
        self.result_path: str | None = None
        self.worker: _Worker | None = None
        self.refresher: _Refresh | None = None
        self.entries = CT.load_catalog()
        self.lang = i18n.lang() if hasattr(i18n, "lang") else "ru"

        lay = QVBoxLayout(self)
        intro = QLabel("Учебные наборы с готовой разметкой: откройте любой и пройдите весь путь - от просмотра разметки до обучения "
                       "и результатов. Всё скачивается без регистрации; из больших наборов берётся только небольшая выборка кадров. "
                       "Файлы сохраняются в папку <code>" + str(paths.examples_dir()) + "</code>.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по названию, классам, описанию…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.task = QComboBox()
        self.task.addItem("Все задачи", "")
        for k, (ru, _en) in CT.KINDS.items():
            self.task.addItem(ru, k)
        self.task.currentIndexChanged.connect(self._filter)
        self.domain = QComboBox()
        self.domain.addItem("Все темы", "")
        for k, v in DOMAINS.items():
            self.domain.addItem(v, k)
        self.domain.currentIndexChanged.connect(self._filter)
        self.count = QLabel("")
        self.count.setStyleSheet("color:#8b92a1")
        self.refresh_btn = QPushButton("Обновить каталог")
        self.refresh_btn.setIcon(icons.icon("refresh"))
        self.refresh_btn.setToolTip("<p>Загружает свежий список наборов из репозитория ML IDE (только по нажатию). "
                                    "Без обновления работает встроенный список.</p>")
        self.refresh_btn.clicked.connect(self._refresh)
        row.addWidget(self.search, 1)
        row.addWidget(self.task)
        row.addWidget(self.domain)
        row.addWidget(self.count)
        row.addWidget(self.refresh_btn)
        lay.addLayout(row)

        sp = QSplitter()
        self.lst = QListWidget()
        self.lst.currentItemChanged.connect(lambda *_: self._show())
        self.lst.itemDoubleClicked.connect(lambda *_: self._run())
        sp.addWidget(self.lst)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.info = QTextBrowser()
        rv.addWidget(self.info, 1)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        rv.addWidget(self.bar)
        rv.addWidget(self.status)
        row = QHBoxLayout()
        self.hf_btn = QPushButton("Свой набор с Hugging Face…")
        self.hf_btn.setIcon(icons.icon("download"))
        self.hf_btn.setToolTip("<p>Введите имя набора вида <b>автор/название</b> (например Francesco/aquarium-qlnqy). Подойдёт любой набор "
                               "с картинками и рамками (колонки image и objects): берётся выборка кадров.</p>")
        self.hf_btn.clicked.connect(self._custom_hf)
        self.go = QPushButton("Скачать и открыть")
        self.go.setProperty("primary", True)
        self.go.clicked.connect(self._run)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        row.addWidget(self.hf_btn)
        row.addStretch(1)
        row.addWidget(close)
        row.addWidget(self.go)
        rv.addLayout(row)
        sp.addWidget(right)
        sp.setSizes([420, 620])
        lay.addWidget(sp, 1)
        self._filter()

    # ---------------------------------------------------------------- список
    def _title(self, e):
        return CT.text_of(e, "title", self.lang)

    def _kind(self, e):
        ru, en = CT.KINDS.get(e.get("task", "detect"), CT.KINDS["detect"])
        return ru if self.lang == "ru" else en

    def _filter(self, *_):
        q = self.search.text().strip().lower()
        task, dom = self.task.currentData(), self.domain.currentData()
        cur = self.current()
        self.lst.blockSignals(True)
        self.lst.clear()
        shown = 0
        for e in self.entries:
            if task and e.get("task") != task:
                continue
            if dom and e.get("domain", "everyday") != dom:
                continue
            hay = " ".join([self._title(e), CT.text_of(e, "what", "ru"), CT.text_of(e, "what", "en"), CT.text_of(e, "title", "ru"),
                            " ".join(map(str, e.get("classes", [])))]).lower()
            if q and q not in hay:
                continue
            nc = len(e.get("classes") or [])
            ready = CT.is_ready(e)
            it = QListWidgetItem(f"{self._title(e)}\n{self._kind(e)}" + (f" · {nc} кл." if nc else "") +
                                 f" · ~{e.get('size_mb', '?')} МБ" + ("  · скачан" if ready else ""))
            it.setData(Qt.UserRole, e["id"])
            self.lst.addItem(it)
            shown += 1
        self.lst.blockSignals(False)
        self.count.setText(f"{shown} из {len(self.entries)}")
        row = 0
        if cur:
            for i in range(self.lst.count()):
                if self.lst.item(i).data(Qt.UserRole) == cur["id"]:
                    row = i
        self.lst.setCurrentRow(row if self.lst.count() else -1)
        self._show()

    def current(self):
        it = self.lst.currentItem()
        if not it:
            return None
        return next((e for e in self.entries if e["id"] == it.data(Qt.UserRole)), None)

    def _show(self):
        e = self.current()
        self.go.setEnabled(e is not None and not (self.worker and self.worker.isRunning()))
        if not e:
            self.info.setHtml("<p style='color:#8b92a1'>Ничего не найдено. Измените поиск или фильтры.</p>")
            return
        ready = CT.is_ready(e)
        self.go.setText("Открыть" if ready else "Скачать и открыть")
        cls = [str(c) for c in (e.get("classes") or [])]
        cls_txt = ", ".join(html.escape(c) for c in cls[:8]) + (f" … +{len(cls) - 8}" if len(cls) > 8 else "") if cls else "по разметке набора"
        fetch = e.get("fetch", {})
        note = ""
        if fetch.get("type") in ("zip_subset", "hf", "hf_rows", "cls_subset") or fetch.get("subset") or fetch.get("per_class"):
            n = fetch.get("subset")
            note = (f"<p style='color:#8b92a1'>Скачивается выборка: до {n} кадров, а не весь набор.</p>" if n else
                    "<p style='color:#8b92a1'>Скачивается небольшая выборка кадров, а не весь набор.</p>")
        learn = CT.text_of(e, "learn", self.lang)
        self.info.setHtml(
            f"<h2>{html.escape(self._title(e))}</h2>"
            f"<p><b>{self._kind(e)}</b> · классы: {cls_txt} · загрузка ~{e.get('size_mb', '?')} МБ"
            f"{' · ' + html.escape(DOMAINS[e['domain']]) if e.get('domain') in DOMAINS else ''}</p>"
            f"<p>{html.escape(CT.text_of(e, 'what', self.lang))}</p>"
            + (f"<p><b>Чему научит:</b> {html.escape(learn)}</p>" if learn else "") + note +
            f"<p style='color:#8b92a1'>Источник: {SOURCES.get(e.get('source', ''), e.get('source', ''))} · "
            f"лицензия данных: {html.escape(str(e.get('license', '')))}</p>")

    # ---------------------------------------------------------------- действия
    def _run(self):
        e = self.current()
        if not e or (self.worker and self.worker.isRunning()):
            return
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.status.setText("")
        self.worker = _Worker(e)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, stage, done, total):
        if total:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
            if total > 100_000:                                   # байты (загрузка zip)
                self.status.setText(f"{stage}: {done / 1048576:.1f} / {total / 1048576:.1f} МБ")
            else:                                                 # кадры
                self.status.setText(f"{stage}: {done} / {total}")
        else:
            self.bar.setRange(0, 0)
            self.status.setText(stage + "…")

    def _done(self, path):
        self.result_path = path
        self.accept()

    def _failed(self, msg):
        self.bar.setVisible(False)
        self._show()
        self.status.setText(f"<span style='color:#ff6b6b'>{html.escape(msg)}</span>")

    def _refresh(self):
        self.refresh_btn.setEnabled(False)
        self.status.setText("Загрузка каталога…")
        self.refresher = _Refresh()
        self.refresher.ok.connect(self._refreshed)
        self.refresher.failed.connect(lambda m: (self.refresh_btn.setEnabled(True),
                                                 self.status.setText(f"<span style='color:#ff6b6b'>Не удалось обновить каталог: {html.escape(m)}</span>")))
        self.refresher.start()

    def _refreshed(self, n):
        self.refresh_btn.setEnabled(True)
        self.entries = CT.load_catalog()
        self.status.setText(f"Каталог обновлён: {len(self.entries)} наборов.")
        self._filter()

    def _custom_hf(self):
        name, ok = QInputDialog.getText(self, "Свой набор с Hugging Face", "Имя набора (автор/название):")
        name = (name or "").strip().strip("/")
        if not ok or not name:
            return
        name = re.sub(r"^https?://huggingface\.co/datasets/", "", name)
        if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", name):
            QMessageBox.warning(self, "Свой набор", "Нужно имя вида автор/название, например Francesco/aquarium-qlnqy.")
            return
        eid = "hf_custom_" + re.sub(r"[^\w]+", "_", name, flags=re.A).lower()
        if not any(e["id"] == eid for e in self.entries):
            self.entries.append({"id": eid, "task": "detect", "domain": "everyday", "source": "huggingface", "license": "см. страницу набора",
                                 "title": {"ru": name, "en": name}, "size_mb": "?", "classes": [],
                                 "what": {"ru": "Набор, добавленный вручную: картинки и рамки (image + objects).",
                                          "en": "A dataset added manually: pictures and boxes (image + objects)."},
                                 "fetch": {"type": "hf_rows", "dataset": name, "subset": 200}})
        self.search.clear()
        self.task.setCurrentIndex(0)
        self.domain.setCurrentIndex(0)
        self._filter()
        for i in range(self.lst.count()):
            if self.lst.item(i).data(Qt.UserRole) == eid:
                self.lst.setCurrentRow(i)

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().reject()
