"""Пути к ресурсам (работает и из исходников, и из собранного .exe)."""
from __future__ import annotations

import sys
from pathlib import Path


def resource(rel: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / rel


def examples_dir() -> Path:
    """Куда скачиваются примеры (папка пользователя, вне репозитория)."""
    return Path.home() / "ML IDE" / "examples"


def child_env() -> dict:
    """Окружение для запуска пользовательского Python (обучение, pip, экспорт модели).

    Собранное приложение (PyInstaller) кладёт свои библиотеки (_internal: msvcp140.dll старой версии, python313.dll и др.)
    в окружение процесса. Дочерний Python из системы подхватывает их вместо своих и аварийно падает (0xC0000005).
    Поэтому убираем из окружения всё, что указывает на папку приложения."""
    import os
    env = dict(os.environ)
    if not getattr(sys, "frozen", False):
        return env
    bad = {os.path.normcase(os.path.abspath(p)) for p in (getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)) if p}

    def keep(part: str) -> bool:
        if not part.strip():
            return False
        a = os.path.normcase(os.path.abspath(part))
        return not any(a == b or a.startswith(b + os.sep) for b in bad)

    for key in ("PATH", "PYTHONPATH"):
        if key in env:
            env[key] = os.pathsep.join(p for p in env[key].split(os.pathsep) if keep(p))
    for key in [k for k in env if k.startswith(("_PYI", "PYSIDE6_", "SHIBOKEN_"))]:
        env.pop(key, None)
    for key in ("PYTHONHOME", "PYTHONNOUSERSITE", "_MEIPASS2", "PYI_APPLICATION_HOME_DIR", "QT_PLUGIN_PATH", "QML2_IMPORT_PATH",
                "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env.pop(key, None)
    return env


def child_qenv():
    """То же для QProcess (QProcessEnvironment)."""
    from PySide6.QtCore import QProcessEnvironment
    q = QProcessEnvironment()
    for k, v in child_env().items():
        q.insert(k, v)
    return q


class child_launch:
    """Контекст на время СТАРТА дочернего Python: сбрасывает «каталог DLL» процесса (SetDllDirectory), который
    выставляет загрузчик PyInstaller (папка _internal со старой msvcp140.dll). Иначе дочерний процесс берёт эти DLL
    и падает с кодом 0xC0000005. После старта значение возвращается обратно."""

    def __enter__(self):
        self._k = None
        if sys.platform == "win32" and getattr(sys, "frozen", False):
            import ctypes
            self._k = ctypes.windll.kernel32
            self._k.SetDllDirectoryW(None)
        return self

    def __exit__(self, *exc):
        if self._k is not None:
            self._k.SetDllDirectoryW(str(getattr(sys, "_MEIPASS", "")) or None)
        return False
