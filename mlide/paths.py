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
