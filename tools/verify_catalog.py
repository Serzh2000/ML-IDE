"""Проверка каталога: для каждой записи скачивает МИНИМАЛЬНУЮ выборку (несколько кадров) и собирает проект.
Не для CI (нужен интернет, идёт долго).  Запуск: python tools/verify_catalog.py report.json [--skip-big] [id ...]"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlide import catalog as C  # noqa: E402
from mlide import paths  # noqa: E402
from mlide.model import Project  # noqa: E402


def check(entry, home):
    e = copy.deepcopy(entry)
    f = e.setdefault("fetch", {})
    if f.get("type") in ("hf", "hf_rows", "zip_subset"):
        f["subset"] = 6
    if f.get("type") in ("cls_zip", "cls_subset"):
        f["per_class"] = 2
    t = time.time()
    try:
        p = C.build_project(e, lambda *a: True)
        pr = Project.load(p)
        kinds = sorted({a.kind for im in pr.images for a in im.anns})
        tags = sum(1 for im in pr.images if im.tag is not None)
        ok = len(pr.images) > 0 and (tags > 0 or kinds)
        return {"id": e["id"], "ok": bool(ok), "images": len(pr.images), "classes": len(pr.class_names()), "kinds": kinds, "tags": tags,
                "kpts": len(pr.keypoints), "sec": round(time.time() - t)}
    except BaseException as ex:  # noqa
        return {"id": e["id"], "ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}", "sec": round(time.time() - t)}


def main():
    out = sys.argv[1]
    args = sys.argv[2:]
    skip_big = "--skip-big" in args
    ids = [a for a in args if not a.startswith("--")]
    home = Path(tempfile.mkdtemp(prefix="mlide_verify_"))
    paths.examples_dir = lambda: home / "examples"
    entries = [e for e in C.load_catalog() if not e.get("legacy")]
    if ids:
        entries = [e for e in entries if e["id"] in ids]
    if skip_big:
        entries = [e for e in entries if not (e.get("fetch", {}).get("type") in ("zip", "cls_zip") and (e.get("size_mb") or 0) > 40)]
    res = []
    done = {}
    if Path(out).exists():                           # продолжаем с места остановки: уже успешные записи не качаем заново
        done = {r["id"]: r for r in json.loads(Path(out).read_text(encoding="utf-8")) if r.get("ok")}
    todo = [e for e in entries if e["id"] not in done]
    res = list(done.values())
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(lambda e: check(e, home), todo):
            res.append(r)
            json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(("OK  " if r["ok"] else "FAIL"), r["id"], {k: v for k, v in r.items() if k not in ("id", "ok")}, flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ok:", sum(r["ok"] for r in res), "of", len(res))
    shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
