"""Английский интерфейс: словарь русских фрагментов -> английский + перехват Qt-функций, задающих текст.

Исходные строки в коде русские. При языке "en" любая строка, попадающая в виджет (подписи, кнопки, подсказки, меню,
таблицы, диалоги, строка состояния, HTML-карточки), переводится по словарю assets/i18n_en.json: сначала самые длинные
фрагменты, на границах слов. Текст, которого нет в словаре, остаётся русским - ничего не ломается.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from . import paths

CYR = re.compile(r"[А-Яа-яЁё]")
_CYR_CLASS = "А-Яа-яЁё"
_lang = "ru"
_pattern: re.Pattern | None = None
_table: dict[str, str] = {}
_hooked = False


def lang() -> str:
    return _lang


def detect_language() -> str:
    env = os.environ.get("MLIDE_LANG")
    if env in ("ru", "en"):
        return env
    try:
        from . import settings
        v = settings.get().value("lang", "ru")
        return v if v in ("ru", "en") else "ru"
    except Exception:
        return "ru"


def save_language(code: str):
    from . import settings
    settings.get().setValue("lang", code)


def load_table(path=None) -> dict[str, str]:
    p = path or paths.resource("assets/i18n_en.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def set_language(code: str, table: dict[str, str] | None = None):
    """Включает язык. Для 'en' строит регулярное выражение по словарю и ставит перехватчики Qt."""
    global _lang, _pattern, _table
    _lang = code
    T.cache_clear()
    if code != "en":
        return
    _table = table if table is not None else load_table()
    pieces = []
    for k in sorted(_table, key=len, reverse=True):
        if not k or not CYR.search(k):
            continue
        lb = f"(?<![{_CYR_CLASS}])" if CYR.match(k[0]) else ""
        la = f"(?![{_CYR_CLASS}])" if CYR.match(k[-1]) else ""
        pieces.append(lb + re.escape(k) + la)
    _pattern = re.compile("|".join(pieces)) if pieces else None
    install_hooks()


@lru_cache(maxsize=20000)
def T(text):
    """Перевод строки (возвращает её же, если язык русский или фрагментов не нашлось)."""
    if _lang != "en" or not isinstance(text, str) or _pattern is None or not CYR.search(text):
        return text
    return _pattern.sub(lambda m: _table[m.group(0)], text)


def tr_list(v):
    if isinstance(v, (list, tuple)):
        return type(v)(T(x) if isinstance(x, str) else x for x in v)
    return v


# ----------------------------------------------------------------------------- перехват Qt
def _tr_args(a, k):
    a = tuple(T(x) if isinstance(x, str) else tr_list(x) if isinstance(x, (list, tuple)) else x for x in a)
    k = {kk: (T(v) if isinstance(v, str) else tr_list(v)) for kk, v in k.items()}
    return a, k


def _wrap_method(cls, name):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_i18n", False):
        return

    def w(self, *a, **k):
        a, k = _tr_args(a, k)
        return orig(self, *a, **k)
    w._i18n = True
    setattr(cls, name, w)


def _wrap_item_text(cls, name):
    """addItem(text, data) / addItem(icon, text, data) / insertItem(index, text, data): переводится только текст."""
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_i18n", False):
        return

    def w(self, *a, **k):
        a = list(a)
        for i, x in enumerate(a):
            if isinstance(x, str):
                a[i] = T(x)
                break                      # первая строка - текст; следующая (если есть) - пользовательские данные
        return orig(self, *a, **k)
    w._i18n = True
    setattr(cls, name, w)


_STATIC_TEXT_ARGS = {          # какие позиционные аргументы статических диалогов - видимый текст (остальное - пути/значения)
    "question": (1, 2), "warning": (1, 2), "information": (1, 2), "critical": (1, 2),
    "getText": (1, 2), "getInt": (1, 2), "getDouble": (1, 2), "getItem": (1, 2),
    "getOpenFileName": (1, 3), "getOpenFileNames": (1, 3), "getSaveFileName": (1, 3), "getExistingDirectory": (1,),
}


def _wrap_static(cls, name):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_i18n", False):
        return
    idx = _STATIC_TEXT_ARGS.get(name, (1, 2))

    def w(*a, **k):
        a = list(a)
        for i in idx:
            if i < len(a) and isinstance(a[i], str):
                a[i] = T(a[i])
        for key in ("caption", "title", "label", "filter"):
            if isinstance(k.get(key), str):
                k[key] = T(k[key])
        return orig(*a, **k)
    w._i18n = True
    setattr(cls, name, staticmethod(w))


def _unused_wrap_static(cls, name):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_i18n", False):
        return

    def w(*a, **k):
        a, k = _tr_args(a, k)
        return orig(*a, **k)
    w._i18n = True
    setattr(cls, name, staticmethod(w))


def _wrap_init(cls):
    orig = cls.__init__
    if getattr(orig, "_i18n", False):
        return

    def w(self, *a, **k):
        a, k = _tr_args(a, k)
        orig(self, *a, **k)
    w._i18n = True
    cls.__init__ = w


def install_hooks():
    global _hooked
    if _hooked:
        return
    _hooked = True
    from PySide6 import QtGui, QtWidgets as W
    methods = {
        W.QLabel: ["setText"], W.QAbstractButton: ["setText"], W.QGroupBox: ["setTitle"],
        W.QWidget: ["setToolTip", "setWindowTitle", "setStatusTip"], W.QLineEdit: ["setPlaceholderText", "setText"],
        W.QTextEdit: ["setHtml", "setText", "setPlainText"], W.QTextBrowser: ["setHtml", "setText", "setPlainText"],
        W.QPlainTextEdit: ["setPlaceholderText", "setPlainText"], W.QComboBox: ["addItems", "setItemText"],
        W.QTabWidget: ["addTab", "setTabText", "insertTab"], W.QListWidget: ["addItem", "addItems"],
        W.QListWidgetItem: ["setText", "setToolTip"], W.QTableWidgetItem: ["setText", "setToolTip"],
        W.QTableWidget: ["setHorizontalHeaderLabels", "setVerticalHeaderLabels"], W.QTreeWidget: ["setHeaderLabels"],
        W.QTreeWidgetItem: ["setText", "setToolTip"], W.QMenu: ["addAction", "addMenu", "setTitle"], W.QMenuBar: ["addMenu"],
        W.QStatusBar: ["showMessage"], W.QFormLayout: ["addRow"], W.QProgressDialog: ["setLabelText"],
        W.QProgressBar: ["setFormat"], W.QDialogButtonBox: ["addButton"], W.QDockWidget: ["setWindowTitle"], W.QGraphicsSimpleTextItem: ["setText"],
        QtGui.QAction: ["setText", "setToolTip", "setIconText", "setStatusTip"], QtGui.QPainter: ["drawText"],
        W.QMessageBox: ["setText", "setInformativeText", "setWindowTitle"],
    }
    for cls, names in methods.items():
        for n in names:
            try:
                _wrap_method(cls, n)
            except Exception:
                pass
    for n in ("addItem", "insertItem"):
        _wrap_item_text(W.QComboBox, n)
    for cls in (W.QMessageBox, W.QInputDialog, W.QFileDialog):
        for n in ("question", "warning", "information", "critical", "getText", "getInt", "getDouble", "getItem",
                  "getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory"):
            try:
                if hasattr(cls, n):
                    _wrap_static(cls, n)
            except Exception:
                pass
    for cls in (W.QLabel, W.QPushButton, W.QCheckBox, W.QGroupBox, W.QToolBar, W.QDockWidget, W.QListWidgetItem,
                W.QTableWidgetItem, W.QTreeWidgetItem, W.QGraphicsSimpleTextItem, W.QProgressDialog, QtGui.QAction,
                W.QRadioButton, W.QToolButton, W.QMenu, W.QPlainTextEdit):
        try:
            _wrap_init(cls)
        except Exception:
            pass
