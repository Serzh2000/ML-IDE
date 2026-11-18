"""Настройки пользователя (язык, тема, недавние проекты, «тур показан»).

Для тестов и скриншотов можно задать MLIDE_SETTINGS_FILE=путь.ini - тогда реальный реестр пользователя не затрагивается.
"""
from __future__ import annotations

import os


def get():
    from PySide6.QtCore import QSettings
    f = os.environ.get("MLIDE_SETTINGS_FILE")
    if f:
        return QSettings(f, QSettings.IniFormat)
    from . import paths
    pr = paths.portable_root()
    if pr is not None:
        pr.mkdir(parents=True, exist_ok=True)
        return QSettings(str(pr / "settings.ini"), QSettings.IniFormat)
    return QSettings("MLIDE", "MLIDE")
