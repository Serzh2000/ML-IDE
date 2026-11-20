"""Запуск приложения целиком (mlide.ui.main_window.run): окно создаётся, приложение закрывается без ошибок и записей об исключениях в журнале."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    work = Path(tempfile.mkdtemp())
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "MLIDE_EXIT_AFTER_MS": "1200", "ML_IDE_NO_TOUR": "1",
           "MLIDE_PORTABLE_DIR": str(work / "data"), "PYTHONIOENCODING": "utf-8", "MLIDE_LANG": "ru"}
    for lang in ("ru", "en"):
        env["MLIDE_LANG"] = lang
        r = subprocess.run([sys.executable, "-m", "mlide"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, (lang, r.returncode, r.stderr[-800:])
        assert "Traceback" not in r.stderr, (lang, r.stderr[-800:])
    log = work / "data" / "logs" / "mlide.log"
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "ERROR" not in text, text[-800:]
    print("ok startup ru/en")
    print("ALL OK")


if __name__ == "__main__":
    main()
