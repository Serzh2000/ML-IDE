"""Портативный режим (настройки и примеры рядом с программой) и выбор языка по умолчанию."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.pop("MLIDE_SETTINGS_FILE", None)
os.environ.pop("MLIDE_LANG", None)
from mlide import i18n, paths, settings  # noqa: E402


def test_portable():
    root = Path(tempfile.mkdtemp()) / "data"
    os.environ["MLIDE_PORTABLE_DIR"] = str(root)
    try:
        assert paths.portable_root() == root
        assert paths.examples_dir() == root / "examples"
        s = settings.get()
        s.setValue("lang", "en")
        s.sync()
        assert (root / "settings.ini").exists(), "настройки должны лежать в data/settings.ini"
        assert i18n.detect_language() == "en"
        s.setValue("lang", "ru")
        s.sync()
        assert i18n.detect_language() == "ru"
    finally:
        del os.environ["MLIDE_PORTABLE_DIR"]
    assert paths.portable_root() is None
    print("ok portable")


def test_default_language():
    root = Path(tempfile.mkdtemp()) / "data"
    os.environ["MLIDE_PORTABLE_DIR"] = str(root)               # пустые настройки: язык берётся по системе
    try:
        got = i18n.detect_language()
        assert got in ("ru", "en") and got == i18n.system_language()
        os.environ["MLIDE_LANG"] = "en"
        assert i18n.detect_language() == "en", "MLIDE_LANG сильнее всего"
    finally:
        os.environ.pop("MLIDE_LANG", None)
        del os.environ["MLIDE_PORTABLE_DIR"]
    print("ok language")


if __name__ == "__main__":
    test_portable()
    test_default_language()
    print("ALL OK")
