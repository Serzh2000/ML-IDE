"""Каталог учебных датасетов: встроенный список + (по кнопке) обновление из репозитория, скачивание без регистрации.

Источники (все публичные, без аккаунта и ключей):
  * Ultralytics assets на GitHub (zip) - из больших архивов берём только нужное число кадров (HTTP Range);
  * Hugging Face Datasets - файлы YOLO-датасета качаются выборочно (картинка + метка), без архивов на гигабайты.
Запись каталога (JSON): id, task (detect|seg|obb|pose|cls), title/what/learn {ru,en}, license, size_mb, classes,
fetch {type: zip|zip_subset|hf, url|repo, subset}.
"""
from __future__ import annotations

import io
import json
import posixpath
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import paths
from . import examples as EX
from . import textio
from .formats import find_label_file, import_yolo_labels
from .importers import import_class_folders
from .model import Project

UA = {"User-Agent": "mlide-catalog"}
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CATALOG_URL = "https://raw.githubusercontent.com/Serzh2000/ML-IDE/main/assets/datasets_catalog.json"
KINDS = {"detect": ("Детекция", "Detection"), "seg": ("Сегментация", "Segmentation"), "obb": ("Повёрнутые рамки", "Rotated boxes"),
         "pose": ("Ключевые точки", "Keypoints"), "cls": ("Классификация", "Classification")}
TASK_OF_KIND = {"Детекция": "detect", "Сегментация": "seg", "Повёрнутые рамки": "obb", "Ключевые точки": "pose", "Классификация": "cls"}


class CatalogError(Exception):
    pass


# --------------------------------------------------------------------------- список
def _bundled():
    p = paths.resource("assets/datasets_catalog.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 0, "entries": []}


def _remote_cache() -> Path:
    return paths.examples_dir().parent / "catalog_remote.json"


def load_catalog() -> list[dict]:
    """Записи каталога: встроенные + (если обновляли) более новые из кэша; затем прежние примеры без дублей."""
    base = _bundled()
    entries, version = base.get("entries", []), base.get("version", 0)
    try:
        cached = json.loads(_remote_cache().read_text(encoding="utf-8"))
        if cached.get("version", 0) > version:
            entries = cached["entries"]
    except Exception:
        pass
    seen, out = set(), []
    for e in entries:
        if e.get("id") and e["id"] not in seen:
            seen.add(e["id"])
            out.append(e)
    for e in EX.EXAMPLES:                        # прежние 9 примеров: проверены, их проекты уже могут быть скачаны
        if e["id"] not in seen:
            out.append(_from_legacy(e))
            seen.add(e["id"])
    return out


def _from_legacy(e: dict) -> dict:
    task = {"obb": "obb", "pose": "pose", "cls": "cls"}.get(e.get("task", ""), "seg" if e["kind"] == "Сегментация" else "detect")
    return {"id": e["id"], "legacy": True, "task": task, "title": {"ru": e["title"], "en": e.get("title_en", e["title"])},
            "what": {"ru": e["what"], "en": e.get("what_en", e["what"])}, "learn": {"ru": e["learn"], "en": e.get("learn_en", e["learn"])},
            "license": "AGPL-3.0", "size_mb": e["size_mb"], "classes": list(e["names"]) if not e.get("keep_top") else [],
            "source": "ultralytics", "level": e["level"]}


def refresh_remote(timeout: int = 20) -> int:
    """Скачивает свежий каталог (по кнопке). Возвращает число записей; ошибки пробрасывает."""
    req = urllib.request.Request(CATALOG_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    if not isinstance(data.get("entries"), list) or not data["entries"]:
        raise CatalogError("Каталог пуст или повреждён")
    _remote_cache().parent.mkdir(parents=True, exist_ok=True)
    _remote_cache().write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return len(data["entries"])


def text_of(e: dict, field: str, lang: str) -> str:
    v = e.get(field, "")
    if isinstance(v, dict):
        return v.get(lang) or v.get("ru") or v.get("en") or ""
    return str(v)


def project_path(e: dict) -> Path:
    eid = str(e["id"])
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", eid):              # id идёт в имя папки: «../x» из чужого каталога недопустим
        raise CatalogError(f"Недопустимый идентификатор набора: {eid!r}")
    return paths.examples_dir() / eid / f"{eid}.mlide.json"


def is_ready(e: dict) -> bool:
    return project_path(e).exists()


# --------------------------------------------------------------------------- сеть
def _open(url, headers=None, timeout=60):
    from . import netsafe
    last = None
    for i in range(4):
        try:
            return netsafe.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=timeout)
        except urllib.error.HTTPError as ex:
            last = ex
            if ex.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as ex:
            last = ex
            if "неразрешён" in str(ex.reason) or "перенаправление" in str(ex.reason):
                raise CatalogError(f"Источник отклонён политикой безопасности: {ex.reason}")
        except Exception as ex:  # noqa
            last = ex
        time.sleep(2 * (i + 1))
    raise CatalogError(f"Не удалось связаться с сервером: {netsafe.explain(last)}.")


MAX_FILE = 300 * 1024 * 1024           # один файл выборки больше 300 МБ не принимаем


def _get_bytes(url, headers=None, timeout=60, max_bytes: int = MAX_FILE) -> bytes:
    """Скачивает файл целиком (с ограничением размера); обрыв посреди чтения повторяется."""
    from . import netsafe
    last = None
    for i in range(4):
        try:
            with _open(url, headers, timeout) as r:
                out, total = [], 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise CatalogError("Файл слишком большой для выборки: загрузка отменена")
                    out.append(chunk)
                return b"".join(out)
        except (urllib.error.HTTPError, CatalogError):
            raise
        except Exception as ex:  # noqa
            last = ex
            time.sleep(1.5 * (i + 1))
    raise CatalogError(f"Не удалось скачать файл: {netsafe.explain(last)}.")


class RangeFile(io.RawIOBase):
    """Файл поверх HTTP Range: zipfile читает оглавление и отдельные файлы большого архива, не качая его целиком."""

    def __init__(self, url):
        self.pos = 0
        with _open(url, {"Range": "bytes=0-0"}) as r:
            self.url = r.geturl()                       # итоговый адрес после перенаправлений (GitHub -> CDN)
            cr = r.headers.get("Content-Range", "")
        if "/" not in cr:
            raise CatalogError("Сервер не поддерживает частичную загрузку")
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
        data = _get_bytes(self.url, {"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


# --------------------------------------------------------------------------- выбор пар «картинка + метка»
def _norm_parts(p: str):
    return posixpath.normpath(p).split("/")


def pair_files(names: list[str]) -> list[tuple[str, str]]:
    """Пары (картинка, метка): метка лежит в соседней папке labels (вместо images) или рядом с тем же именем."""
    nameset = set(names)
    pairs = []
    for n in names:
        if not n.lower().endswith(IMG):
            continue
        base = posixpath.splitext(n)[0]
        parts = _norm_parts(n)
        cands = [base + ".txt"]
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].lower() == "images":
                cands.insert(0, posixpath.splitext("/".join(parts[:i] + ["labels"] + parts[i + 1:]))[0] + ".txt")
                break
        lab = next((c for c in cands if c in nameset), None)
        if lab:
            pairs.append((n, lab))
    if pairs:
        return pairs
    # запасной вариант: сопоставление по имени файла (разные структуры папок)
    labs = {posixpath.splitext(posixpath.basename(n))[0]: n for n in names if n.lower().endswith(".txt") and "label" in n.lower()}
    for n in names:
        if n.lower().endswith(IMG):
            k = posixpath.splitext(posixpath.basename(n))[0]
            if k in labs:
                pairs.append((n, labs[k]))
    return pairs


def _choose(pairs, limit, seed=0):
    pairs = sorted(pairs)
    random.Random(seed).shuffle(pairs)
    return pairs[:limit] if limit else pairs


def _safe_name(s: str) -> str:
    """Имя файла/папки без символов, недопустимых в Windows; кириллица и прочий юникод сохраняются."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(s)).strip().rstrip(". ")
    return s[:60] or "x"


# --------------------------------------------------------------------------- загрузчики
def _read_member(url: str, zi: zipfile.ZipInfo) -> bytes:
    """Один файл из удалённого zip одним запросом: локальный заголовок + сжатые данные, потом распаковка."""
    import struct
    import zlib
    want = 30 + 1024 + zi.compress_size                    # заголовок (30) + имя/extra (обычно < 1 КБ) + данные
    raw = _get_bytes(url, {"Range": f"bytes={zi.header_offset}-{zi.header_offset + want - 1}"}, timeout=120)
    sig, _v, _f, method, _t, _d, _crc, csize, _usize, fn, ex = struct.unpack("<4s5H3L2H", raw[:30])
    if sig != b"PK":
        raise CatalogError("Повреждённый архив")
    start = 30 + fn + ex
    if start + zi.compress_size > len(raw):                # редкий случай: длинное имя/extra - докачиваем недостающее
        raw += _get_bytes(url, {"Range": f"bytes={zi.header_offset + len(raw)}-{zi.header_offset + start + zi.compress_size - 1}"}, timeout=120)
    data = raw[start:start + zi.compress_size]
    if method == 0:
        return data
    if method == 8:
        if zi.file_size > MAX_FILE:
            raise CatalogError("Файл в архиве слишком большой для выборки")
        dec = zlib.decompressobj(-15)
        out = dec.decompress(data, zi.file_size + 1)                  # распаковка с потолком: «бомба» не раздует память
        if dec.unconsumed_tail or len(out) > zi.file_size:
            raise CatalogError("Повреждённый или подозрительный архив: размер файла не совпадает с заголовком")
        return out
    raise CatalogError(f"Неподдерживаемое сжатие в архиве ({method})")


def _fetch_zip_subset(url: str, raw: Path, limit: int, progress):
    """Из большого zip извлекаем только limit пар картинка+метка и yaml-файл (по одному запросу на файл, параллельно)."""
    progress("Чтение оглавления архива", 0, 0)
    rf = RangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(rf, 1 << 18))
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    pairs = _choose(pair_files(names), limit)
    if not pairs:
        raise CatalogError("В архиве не нашлось пар «картинка + метка»")
    extra = [n for n in names if n.lower().endswith((".yaml", ".yml")) and z.getinfo(n).file_size < 200_000][:2]
    total = len(pairs)
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "labels").mkdir(parents=True, exist_ok=True)
    state = {"done": 0, "stop": False}

    def one(item):
        k, (img, lab) = item
        if state["stop"]:
            return
        stem = f"{k:05d}_{_safe_name(posixpath.splitext(posixpath.basename(img))[0])}"
        (raw / "images" / (stem + posixpath.splitext(img)[1].lower())).write_bytes(_read_member(rf.url, z.getinfo(img)))
        (raw / "labels" / (stem + ".txt")).write_bytes(_read_member(rf.url, z.getinfo(lab)))
        state["done"] += 1
        if progress("Загрузка кадров", state["done"], total) is False:
            state["stop"] = True

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, list(enumerate(pairs))))
    if state["stop"]:
        raise CatalogError("Отменено")
    for y in extra:
        (raw / posixpath.basename(y)).write_bytes(_read_member(rf.url, z.getinfo(y)))


def _fetch_hf(repo: str, raw: Path, limit: int, progress):
    """Файлы датасета Hugging Face: берём limit пар картинка+метка и data.yaml, остальное не качаем."""
    progress("Чтение списка файлов", 0, 0)
    files, url = [], f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&limit=1000"
    for _ in range(80):
        with _open(url, timeout=180) as r:
            files += [x for x in json.loads(r.read().decode("utf-8")) if x.get("type") == "file"]
            link = r.headers.get("Link", "")
        nxt = next((p.split(";")[0].strip().strip("<>") for p in link.split(",") if 'rel="next"' in p), None)
        if not nxt:
            break
        url = nxt
    names = [f["path"] for f in files]
    pairs = _choose(pair_files(names), limit)
    if not pairs:
        raise CatalogError("В репозитории не нашлось пар «картинка + метка»")
    ydoc = next((n for n in names if n.lower().endswith(("data.yaml", "dataset.yaml", "data.yml"))), None)
    base = f"https://huggingface.co/datasets/{repo}/resolve/main/"
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "labels").mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, (img, lab) in enumerate(pairs):
        stem = f"{k:05d}_{_safe_name(posixpath.splitext(posixpath.basename(img))[0])}"
        jobs.append((base + urllib.parse.quote(img), raw / "images" / (stem + posixpath.splitext(img)[1].lower())))
        jobs.append((base + urllib.parse.quote(lab), raw / "labels" / (stem + ".txt")))
    if ydoc:
        jobs.append((base + urllib.parse.quote(ydoc), raw / "data.yaml"))
    state = {"done": 0, "stop": False}

    def one(job):
        if state["stop"]:
            return
        u, dest = job
        dest.write_bytes(_get_bytes(u, timeout=120))
        state["done"] += 1
        if progress("Загрузка кадров", state["done"] // 2, len(pairs)) is False:
            state["stop"] = True

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    if state["stop"]:
        raise CatalogError("Отменено")


def _fetch_cls_subset(url: str, raw: Path, per_class: int, progress):
    """Классификация: из удалённого zip с папками train/<класс>/ берём по per_class картинок каждого класса."""
    progress("Чтение оглавления архива", 0, 0)
    rf = RangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(rf, 1 << 18))
    split = {"train", "val", "valid", "validation", "test", "training"}
    by: dict[str, list[str]] = {}
    for i in z.infolist():
        n = i.filename
        if i.is_dir() or not n.lower().endswith(IMG):
            continue
        parts = _norm_parts(n)
        if len(parts) < 2 or parts[-2].lower() in split:
            continue
        by.setdefault(parts[-2], []).append(n)
    if not by:
        raise CatalogError("В архиве не найдены папки с классами")
    want = []
    rnd = random.Random(0)
    for cls in sorted(by):
        files = sorted(by[cls])
        rnd.shuffle(files)
        want += [(cls, f) for f in files[:per_class]]
    total, state = len(want), {"done": 0, "stop": False}

    def one(item):
        cls, f = item
        if state["stop"]:
            return
        d = raw / _safe_name(cls)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{_safe_name(posixpath.basename(f))}").write_bytes(_read_member(rf.url, z.getinfo(f)))
        state["done"] += 1
        if progress("Загрузка кадров", state["done"], total) is False:
            state["stop"] = True

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, want))
    if state["stop"]:
        raise CatalogError("Отменено")


def _find_objects(feats: list):
    """Имена классов из описания колонок датасета (ClassLabel внутри objects)."""
    names = None
    for f in feats:
        t = f.get("type", {})
        if f["name"] in ("objects", "annotations", "labels") and isinstance(t, dict):
            cat = t.get("category") or t.get("categories") or t.get("label") or t.get("labels") or t.get("category_id")
            if isinstance(cat, dict):
                inner = cat.get("feature", cat)
                if isinstance(inner, dict) and inner.get("names"):
                    names = [str(x) for x in inner["names"]]
    return names


def _row_boxes(row: dict, w: int, h: int):
    """[(класс, cx, cy, bw, bh)] в долях кадра из строки датасета (формат COCO xywh в пикселях)."""
    obj = row.get("objects")
    out = []
    items = []
    if isinstance(obj, dict):
        bb = obj.get("bbox") or obj.get("bboxes") or []
        cl = obj.get("category") or obj.get("categories") or obj.get("label") or obj.get("labels") or obj.get("category_id") or []
        items = list(zip(bb, cl))
    elif isinstance(obj, list):
        for o in obj:
            if isinstance(o, dict) and o.get("bbox") is not None:
                items.append((o["bbox"], o.get("category", o.get("category_id", o.get("label", 0)))))
    for bb, c in items:
        try:
            x, y, bw, bh = [float(v) for v in bb]
            c = int(c)
        except (TypeError, ValueError):
            continue
        if bw <= 1 or bh <= 1 or w <= 0 or h <= 0:
            continue
        x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append((c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return out


def _fetch_hf_rows(dataset: str, raw: Path, limit: int, progress):
    """Датасет Hugging Face «картинка + objects(bbox, category)» через datasets-server (JSON; без pyarrow и без аккаунта)."""
    base = "https://datasets-server.huggingface.co/"
    ds = urllib.parse.quote(dataset, safe="/")
    progress("Чтение описания набора", 0, 0)
    sp = json.loads(_get_bytes(f"{base}splits?dataset={ds}").decode("utf-8"))
    splits = [x["split"] for x in sp.get("splits", [])]
    if not splits:
        raise CatalogError("У набора нет доступных частей (splits)")
    order = [x for x in ("train", "training") if x in splits] + [x for x in splits if x not in ("train", "training")]
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "labels").mkdir(parents=True, exist_ok=True)
    names, tasks = None, []
    for split in order:
        offset = 0
        while len(tasks) < limit:
            n = min(100, limit - len(tasks))
            d = json.loads(_get_bytes(f"{base}rows?dataset={ds}&config=default&split={split}&offset={offset}&length={n}").decode("utf-8"))
            names = names or _find_objects(d.get("features", []))
            rows = d.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r["row"]
                im = row.get("image")
                if not isinstance(im, dict) or not im.get("src"):
                    continue
                w, h = int(row.get("width") or im.get("width") or 0), int(row.get("height") or im.get("height") or 0)
                tasks.append((im["src"], _row_boxes(row, w, h), len(tasks)))
            offset += len(rows)
            if offset >= d.get("num_rows_total", 0):
                break
        if len(tasks) >= limit:
            break
    if not tasks:
        raise CatalogError("В наборе не нашлось картинок с рамками")
    total, state = len(tasks), {"done": 0, "stop": False}
    maxc = {"v": -1}

    def one(t):
        if state["stop"]:
            return
        src, boxes, k = t
        data = _get_bytes(src, timeout=120)
        ext = ".png" if data[:4] == b"\x89PNG" else ".jpg"
        stem = f"{k:05d}"
        (raw / "images" / (stem + ext)).write_bytes(data)
        lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in boxes]
        (raw / "labels" / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        for bx in boxes:
            maxc["v"] = max(maxc["v"], bx[0])
        state["done"] += 1
        if progress("Загрузка кадров", state["done"], total) is False:
            state["stop"] = True

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, tasks))
    if state["stop"]:
        raise CatalogError("Отменено")
    nm = list(names or [])
    while len(nm) <= maxc["v"]:
        nm.append(f"class_{len(nm)}")
    import yaml
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": {i: n for i, n in enumerate(nm)}}, allow_unicode=True), encoding="utf-8")


def _fetch_zip(url: str, dest: Path, progress):
    """Небольшой zip качаем целиком (с проверкой целостности)."""
    EX._download(url, dest, progress)


# --------------------------------------------------------------------------- сборка проекта
def _read_yaml(raw: Path):
    try:
        import yaml
        for y in list(raw.glob("*.yaml")) + list(raw.rglob("data.yaml")):
            cfg = yaml.safe_load(y.read_text(encoding="utf-8", errors="replace"))
            if isinstance(cfg, dict) and cfg.get("names") is not None:
                nm = cfg["names"]
                nm = [nm[k] for k in sorted(nm)] if isinstance(nm, dict) else list(nm)
                return [str(x) for x in nm], cfg
    except Exception:
        pass
    return None, {}


def _infer_names(raw: Path, entry: dict, imgs):
    names = list(entry.get("classes") or [])
    cfg = {}
    ynames, cfg = _read_yaml(raw)
    if ynames and (not names or len(ynames) >= len(names)):
        names = ynames
    mx = -1                                         # классы по самим меткам: хотя бы class_0..N, чтобы ничего не потерять
    for p in imgs[:400]:
        lf = find_label_file(str(p))
        if lf is not None:
            for ln in textio.read_text(lf).splitlines():
                try:
                    mx = max(mx, int(float(ln.split()[0])))
                except (ValueError, IndexError):
                    pass
    while len(names) <= mx:
        names.append(f"class_{len(names)}")
    return names, cfg


def build_project(entry: dict, progress=lambda stage, done, total: True) -> Path:
    """Скачивает набор (если нужно) и собирает проект ML IDE. Возвращает путь к .mlide.json.
    Если сборка не удалась, распакованные данные удаляются: следующая попытка начнёт заново, а не упрётся в ту же ошибку."""
    try:
        return _build_project(entry, progress)
    except BaseException:
        try:
            pf = project_path(entry)
            if not pf.exists():
                shutil.rmtree(pf.parent / "raw", ignore_errors=True)
        except Exception:  # noqa
            pass
        raise


def _build_project(entry: dict, progress) -> Path:
    legacy = EX.BY_ID.get(entry["id"])
    if legacy and entry.get("legacy"):
        return EX.build_example(legacy, progress)
    proj_file = project_path(entry)
    if proj_file.exists():
        return proj_file
    root = proj_file.parent
    raw = root / "raw"
    fetch = entry.get("fetch", {})
    kind, limit = fetch.get("type"), int(fetch.get("subset") or 300)
    if not raw.exists():
        tmp = root / "raw.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "hf_rows":
                _fetch_hf_rows(fetch["dataset"], tmp, limit, progress)
            elif kind == "hf":
                _fetch_hf(fetch["repo"], tmp, limit, progress)
            elif kind == "cls_subset":
                _fetch_cls_subset(fetch["url"], tmp, int(fetch.get("per_class") or 40), progress)
            elif kind == "zip_subset":
                _fetch_zip_subset(fetch["url"], tmp, limit, progress)
            elif kind in ("zip", "cls_zip"):
                zp = root / "download.zip"
                if zp.exists() and not zipfile.is_zipfile(zp):      # остаток прерванной загрузки прежних версий
                    zp.unlink()
                if not zp.exists():
                    _fetch_zip(fetch["url"], zp, progress)
                progress("Распаковка", 0, 0)
                EX._safe_extract(zp, tmp)
                zp.unlink(missing_ok=True)
            else:
                raise CatalogError(f"Неизвестный тип источника: {kind}")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        tmp.rename(raw)

    progress("Сборка проекта", 0, 0)
    task = entry.get("task", "detect")
    title = text_of(entry, "title", "ru")
    if task == "cls":
        pr = Project(title)
        _prune_cls(raw, int(fetch.get("per_class") or 40))
        import_class_folders(pr, raw)
        for e in pr.images:
            e.reviewed = True
        pr.save(proj_file)
        return proj_file
    imgs = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG and "labels" not in p.relative_to(raw).parts)
    random.Random(0).shuffle(imgs)
    imgs = imgs[:limit] if limit and kind == "zip" else imgs
    names, cfg = _infer_names(raw, entry, imgs)
    pr = Project(title)
    for n in names:
        pr.add_class(n)
    pr.add_images(imgs)
    kpt = cfg.get("kpt_shape") or entry.get("kpt_shape")
    n_k, k_dim = (int(kpt[0]), int(kpt[1])) if task == "pose" and kpt else (0, 3)
    if n_k:
        pr.keypoints = list(entry.get("keypoints") or [f"p{i + 1}" for i in range(n_k)])
        pr.skeleton = [list(p) for p in entry.get("skeleton", [])]
        pr.flip_pairs = [list(p) for p in entry.get("flip", [])]
    got = 0
    for e in pr.images:
        got += import_yolo_labels(e, len(names), n_k, k_dim, obb=(task == "obb"))
        e.reviewed = True
    if got == 0:
        raise CatalogError("Не удалось прочитать разметку набора (формат меток не распознан)")
    pr.save(proj_file)
    return proj_file


def _prune_cls(raw: Path, per_class: int):
    """Для учебного примера хватает нескольких десятков картинок на класс: лишнее удаляем до импорта (CIFAR - 60 тысяч файлов)."""
    split = {"train", "val", "valid", "validation", "test", "training"}
    by: dict[str, list[Path]] = {}
    for p in sorted(raw.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG and p.parent != raw:
            name = p.parent.name
            if name.lower() in split:
                continue
            by.setdefault(name, []).append(p)
    rnd = random.Random(0)
    for files in by.values():
        rnd.shuffle(files)
        for p in files[per_class:]:
            p.unlink(missing_ok=True)
