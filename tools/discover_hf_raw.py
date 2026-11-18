"""Поиск на Hugging Face датасетов YOLO, лежащих "россыпью" (images/ + labels/ + data.yaml, без zip).

Для каждого репозитория: список файлов, пары картинка+метка по имени, data.yaml (классы, kpt_shape), тип меток по одной метке.
Результат - JSON-список кандидатов.  Запуск: python tools/discover_hf_raw.py out.json [запрос ...]
"""
from __future__ import annotations

import json
import posixpath
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import yaml

UA = {"User-Agent": "mlide-catalog-builder"}
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _open(req, timeout=30):
    import time
    for i in range(6):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(15 * (i + 1))
    raise RuntimeError("rate limited")


def fetch(url, timeout=30):
    with _open(urllib.request.Request(url, headers=UA), timeout) as r:
        return r.read()


def api(path):
    return json.loads(fetch("https://huggingface.co/api/" + path).decode("utf-8"))


def tree(rid):
    """Все файлы репозитория (с постраничной выдачей)."""
    out, url = [], f"https://huggingface.co/api/datasets/{rid}/tree/main?recursive=true&limit=1000"
    for _ in range(60):
        req = urllib.request.Request(url, headers=UA)
        with _open(req) as r:
            out += json.loads(r.read().decode("utf-8"))
            link = r.headers.get("Link", "")
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split(";")[0].strip().strip("<>")
        if not nxt:
            break
        url = nxt
    return [x for x in out if x.get("type") == "file"]


def probe(ds):
    rid = ds["id"]
    try:
        meta = api(f"datasets/{rid}")
        files = tree(rid)
    except Exception:
        return None
    names = {f["path"]: f.get("size", 0) for f in files}
    imgs = {}
    for p in names:
        if p.lower().endswith(IMG):
            imgs.setdefault(posixpath.splitext(posixpath.basename(p))[0], []).append(p)
    labs = {}
    for p in names:
        if p.lower().endswith(".txt") and "label" in p.lower():
            labs.setdefault(posixpath.splitext(posixpath.basename(p))[0], []).append(p)
    pairs = [k for k in imgs if k in labs]
    if len(pairs) < 40:
        return None
    ydoc = next((p for p in names if p.lower().endswith(("data.yaml", "dataset.yaml", "data.yml"))), None)
    info = {}
    if ydoc:
        try:
            cfg = yaml.safe_load(fetch(f"https://huggingface.co/datasets/{rid}/resolve/main/{urllib.parse.quote(ydoc)}").decode("utf-8", "replace"))
            nm = cfg.get("names")
            if isinstance(nm, dict):
                nm = [nm[k] for k in sorted(nm)]
            info["names"] = list(nm) if nm else None
            info["kpt_shape"] = cfg.get("kpt_shape")
        except Exception:
            pass
    sample = labs[pairs[0]][0]
    cols = 0
    try:
        txt = fetch(f"https://huggingface.co/datasets/{rid}/resolve/main/{urllib.parse.quote(sample)}").decode("utf-8", "replace")
        line = next((ln for ln in txt.splitlines() if ln.strip()), "")
        cols = len(line.split())
    except Exception:
        pass
    img_mb = sum(names[v[0]] for k, v in imgs.items() if k in labs) / 1e6
    lic = next((t.split(":", 1)[1] for t in meta.get("tags", []) if t.startswith("license:")), "")
    return {"repo": rid, "pairs": len(pairs), "images_total_mb": round(img_mb, 0), "yaml": ydoc, "cols": cols, "license": lic,
            "downloads": meta.get("downloads"), "tags": [t for t in meta.get("tags", []) if t.startswith("task_")], **info}


def main():
    out = sys.argv[1]
    queries = sys.argv[2:] or ["yolo", "yolov8", "yolov5", "yolo11", "object-detection", "ultralytics", "yolo segmentation", "yolo obb",
                               "yolo pose", "detection dataset yolo format", "bounding box yolo", "roboflow yolo", "data.yaml"]
    seen = {}
    for q in queries:
        try:
            url = (f"datasets?filter={urllib.parse.quote(q[7:])}&limit=1000&sort=downloads&direction=-1" if q.startswith("filter:")
                   else f"datasets?search={urllib.parse.quote(q)}&limit=100&sort=downloads&direction=-1")
            for x in api(url):
                seen[x["id"]] = x
        except Exception as e:  # noqa
            print("query failed", q, e, file=sys.stderr)
    print("repos:", len(seen), file=sys.stderr)
    res = []
    with ThreadPoolExecutor(5) as ex:
        for r in ex.map(probe, list(seen.values())):
            if r:
                res.append(r)
                print("ok", r["repo"], r["pairs"], r["cols"], r["license"], file=sys.stderr)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("found:", len(res), file=sys.stderr)


if __name__ == "__main__":
    main()
