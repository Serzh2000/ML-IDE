"""Скачивает нужные иконки Lucide (лицензия ISC) в assets/icons: python tools/fetch_icons.py

Имя значка в приложении -> варианты имён в наборе Lucide (в разных версиях часть значков переименована).
"""
import sys
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "1.73.0"
BASE = f"https://cdn.jsdelivr.net/npm/lucide-static@{VERSION}"
OUT = Path(__file__).resolve().parent.parent / "assets" / "icons"

WANTED = {
    # инструменты разметки
    "select": ["mouse-pointer-2", "mouse-pointer"], "box": ["square"], "polygon": ["pentagon"],
    "smart": ["wand-sparkles", "wand-2", "sparkles"], "sam": ["mouse-pointer-click", "pointer"],
    "obb": ["rectangle-horizontal", "rectangle-vertical"], "pose": ["person-standing", "accessibility"], "tag": ["tag"],
    # навигация и действия
    "prev": ["chevron-left"], "next": ["chevron-right"], "reviewed": ["check-check", "check"],
    "auto": ["bot", "cpu"], "accept": ["check"], "augment": ["sliders-horizontal", "sliders"],
    "stats": ["chart-column", "bar-chart-3", "bar-chart"], "advisor": ["lightbulb"], "export": ["package", "upload"],
    "train": ["play"], "results": ["chart-line", "line-chart"], "help": ["circle-help", "help-circle"],
    # стартовый экран и диалоги
    "images": ["images", "image"], "folder": ["folder"], "open": ["folder-open"], "examples": ["graduation-cap"],
    "file": ["file-json", "file-text", "file"], "timer": ["timer"], "stop": ["square"], "start": ["play"],
    "tour": ["route", "compass"],
    # статусы
    "ok": ["circle-check", "check-circle"], "warn": ["triangle-alert", "alert-triangle"], "info": ["info"],
    "circle": ["circle"], "dot": ["circle-dot"], "dashed": ["circle-dashed"], "done": ["circle-check", "check-circle-2"],
    "refresh": ["refresh-cw"], "download": ["download"], "search": ["search"], "model": ["brain"],
    "plus": ["plus"], "minus": ["minus"], "palette": ["palette"], "pencil": ["pencil", "edit-2"], "import": ["file-input", "import"],
    "merge": ["merge"],
    "video": ["video"], "camera": ["camera"], "copy": ["copy"], "trash": ["trash-2", "trash"],
}


def get(name):
    req = urllib.request.Request(f"{BASE}/icons/{name}.svg", headers={"User-Agent": "mlide-icon-fetch"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    for key, candidates in WANTED.items():
        for nm in candidates:
            try:
                svg = get(nm)
            except urllib.error.URLError:
                continue
            (OUT / f"{key}.svg").write_text(svg, encoding="utf-8")
            break
        else:
            missing.append(key)
    try:
        req = urllib.request.Request(f"{BASE}/LICENSE", headers={"User-Agent": "mlide-icon-fetch"})
        (OUT / "LICENSE-lucide.txt").write_text(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"), encoding="utf-8")
    except Exception as e:  # noqa
        print("LICENSE not downloaded:", e)
    print(f"saved {len(WANTED) - len(missing)} icons to {OUT}; missing: {missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
