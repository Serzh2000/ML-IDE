"""Журнал ошибок приложения. В собранном exe нет консоли, поэтому исключения и предупреждения Qt пишутся в файл
(<папка данных>/logs/mlide.log), а пользователь видит сообщение с кнопкой «Открыть папку журнала»."""
from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

_log: logging.Logger | None = None
_fault_file = None
_shown: set = set()


def log_dir() -> Path:
    from . import paths
    pr = paths.portable_root()
    if pr is not None:
        return pr / "logs"
    try:
        from PySide6.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        if base:
            return Path(base) / "ML IDE" / "logs"
    except Exception:  # noqa
        pass
    return Path.home() / "ML IDE" / "logs"


def logger() -> logging.Logger:
    global _log
    if _log is None:
        _log = logging.getLogger("mlide")
        _log.setLevel(logging.INFO)
        try:
            d = log_dir()
            d.mkdir(parents=True, exist_ok=True)
            h = RotatingFileHandler(d / "mlide.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            _log.addHandler(h)
        except Exception:  # noqa
            _log.addHandler(logging.NullHandler())
    return _log


def install(show_dialog=True):
    """Перехват необработанных исключений (в том числе в слотах Qt и потоках), сообщений Qt и аварий C++ (faulthandler)."""
    global _fault_file
    lg = logger()
    lg.info("start")
    try:
        _fault_file = open(log_dir() / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(_fault_file, all_threads=True)
    except Exception:  # noqa
        pass

    def report(exc_type, exc, tb, where=""):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        lg.error("%s\n%s", where or "unhandled exception", text)
        key = f"{exc_type.__name__}:{exc}"
        if show_dialog and key not in _shown and len(_shown) < 5:
            _shown.add(key)
            _notify(f"{exc_type.__name__}: {exc}")

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return sys.__excepthook__(exc_type, exc, tb)
        report(exc_type, exc, tb)

    sys.excepthook = hook
    threading.excepthook = lambda a: report(a.exc_type, a.exc_value, a.exc_traceback, f"thread {getattr(a.thread, 'name', '')}")
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler

        def qt_handler(kind, ctx, msg):
            if kind in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                lg.error("Qt: %s", msg)
            elif kind == QtMsgType.QtWarningMsg and "QThread" in msg:
                lg.warning("Qt: %s", msg)

        qInstallMessageHandler(qt_handler)
    except Exception:  # noqa
        pass


def _notify(msg: str):
    try:
        from PySide6.QtCore import QTimer, QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is None:
            return

        def show():
            box = QMessageBox(QMessageBox.Warning, "Ошибка",
                              "В программе произошла ошибка. Работа продолжается, но последнее действие могло не выполниться.\n\n" +
                              msg[:400] + "\n\nПодробности записаны в журнал.")
            open_btn = box.addButton("Открыть папку журнала", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Close)
            box.exec()
            if box.clickedButton() is open_btn:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir())))

        QTimer.singleShot(0, QApplication.instance(), show)      # выполняется в потоке GUI, даже если исключение пришло из другого потока
    except Exception:  # noqa
        pass


def open_log_folder():
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    d = log_dir()
    d.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))
