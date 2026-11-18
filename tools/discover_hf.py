"""Поиск готовых YOLO-датасетов на Hugging Face (без регистрации): zip-архивы, внутри которых лежат картинки и метки YOLO.

Не лезем внутрь больших файлов целиком: по HTTP Range читаем только оглавление zip. Результат - candidates.json,
из которого вручную/скриптом собирается assets/datasets_catalog.json.

Запуск:  python tools/discover_hf.py candidates.json [запрос ...]
"""
from __future__ import annotations

import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "mlide-catalog-builder"}
MAX_MB = 700
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def api(path):
    with get("https://huggingface.co/api/" + path) as r:
        return json.loads(r.read().decode("utf-8"))


class RangeFile(io.RawIOBase):
    """Файл-обёртка над HTTP Range: zipfile читает только оглавление."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cr = r.headers.get("Content-Range", "")
            self.size = int(cr.split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def inspect_zip(url):
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 16))
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    imgs = [n for n in names if n.lower().endswith(IMG)]
    txts = [n for n in names if n.lower().endswith(".txt") and "/labels/" in "/" + n]
    yamls = [n for n in names if n.lower().endswith((".yaml", ".yml"))]
    info = {"images": len(imgs), "labels": len(txts), "yaml": yamls[:3], "n_files": len(names)}
    data_yaml = next((y for y in yamls if y.lower().endswith("data.yaml")), yamls[0] if yamls else None)
    if data_yaml:
        try:
            import yaml
            cfg = yaml.safe_load(z.read(data_yaml).decode("utf-8", "replace"))
            nm = cfg.get("names")
            if isinstance(nm, dict):
                nm = [nm[k] for k in sorted(nm)]
            info["names"] = list(nm) if nm else None
            info["kpt_shape"] = cfg.get("kpt_shape")
        except Exception:
            pass
    # тип разметки по первой метке
    for t in txts[:30]:
        try:
            line = next((l for l in z.read(t).decode("utf-8", "replace").splitlines() if l.strip()), "")
        except Exception:
            continue
        if line:
            n = len(line.split())
            info["cols"] = n
            break
    return info


def candidates(queries, limit=100):
    seen = {}
    for q in queries:
        try:
            for x in api(f"datasets?search={urllib.parse.quote(q)}&limit={limit}&sort=downloads&direction=-1"):
                seen[x["id"]] = x
        except Exception as e:  # noqa
            print("query failed", q, e, file=sys.stderr)
    return list(seen.values())


def probe(ds):
    rid = ds["id"]
    try:
        meta = api(f"datasets/{rid}?blobs=true")
    except Exception as e:  # noqa
        return None
    lic = next((t.split(":", 1)[1] for t in meta.get("tags", []) if t.startswith("license:")), "")
    zips = [s for s in meta.get("siblings", []) if s["rfilename"].lower().endswith(".zip") and (s.get("size") or 0) <= MAX_MB * 1e6
            and (s.get("size") or 0) > 20_000]
    out = []
    for s in sorted(zips, key=lambda s: s.get("size") or 0)[:3]:
        url = f"https://huggingface.co/datasets/{rid}/resolve/main/{urllib.parse.quote(s['rfilename'])}"
        try:
            info = inspect_zip(url)
        except Exception as e:  # noqa
            continue
        if info["images"] >= 20 and info["labels"] >= 20:
            out.append({"repo": rid, "file": s["rfilename"], "url": url, "size_mb": round((s.get("size") or 0) / 1e6, 1),
                        "license": lic, "downloads": meta.get("downloads"), "tags": [t for t in meta.get("tags", []) if t.startswith("task_")],
                        "card": (meta.get("cardData") or {}).get("pretty_name") or "", **info})
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "candidates.json"
    queries = sys.argv[2:] or ["yolo", "yolov8", "yolov5", "yolo11", "object-detection", "detection yolo format", "obb yolo", "pose yolo",
                               "segmentation yolo", "bounding box", "ultralytics"]
    cands = candidates(queries)
    print("repos:", len(cands), file=sys.stderr)
    res = []
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(probe, cands):
            if r:
                res.extend(r)
                print("ok", r[0]["repo"], len(r), file=sys.stderr)
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("found zips:", len(res), file=sys.stderr)


if __name__ == "__main__":
    main()
