"""YOLO / Ultralytics / Darknet: по одному .txt на картинку, строки «класс + нормированные координаты».

Понимает рамки (4 числа), полигоны сегментации (чётное число >= 6), повёрнутые рамки (8 чисел), ключевые точки (kpt_shape
из data.yaml), рамку с уверенностью (5 чисел). Названия классов берутся из data.yaml / classes.txt / obj.names / names.txt.
Метки могут лежать где угодно: связь с картинкой строится по пути (labels -> images) и по имени файла.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from ..types import Detection, Obj, Parsed, Record

NAME = "yolo"
TITLE = {"ru": "YOLO (txt на каждую картинку)", "en": "YOLO (one txt per image)"}

NAMES_FILES = {"classes.txt", "obj.names", "names.txt", "labels.txt", "classes.names", "class_names.txt", "classes.names.txt"}
NOT_LABELS = NAMES_FILES | {"train.txt", "val.txt", "valid.txt", "test.txt", "readme.txt", "license.txt", "requirements.txt",
                            "train.cache", "notes.txt", "classes.csv"}
YAML_NAMES = {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}


def _numbers(tokens):
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            return None
    return out


def _line_kind(vals: list[float], kpt: tuple | None):
    """Вид строки по числу значений после класса: box | box_conf | poly | obb | pose | None."""
    n = len(vals)
    if n == 4:
        return "box"
    if kpt and n == 4 + kpt[0] * kpt[1]:
        return "pose"
    if n == 5:
        return "box_conf"
    if n == 8:
        return "obb"
    if n >= 6 and n % 2 == 0:
        return "poly"
    return None


def _candidates(ctx):
    return [f for f in ctx.files(".txt") if f.name.lower() not in NOT_LABELS]


def _read_cfg(ctx):
    """data.yaml: names, kpt_shape. -> (names, kpt_shape, yaml_path)."""
    import yaml
    for f in ctx.all_files():
        if f.suffix.lower() in (".yaml", ".yml") and (f.name.lower() in YAML_NAMES or f.stat().st_size < 200_000):
            try:
                cfg = yaml.safe_load(ctx.read_text(f))
            except Exception:  # noqa
                continue
            if isinstance(cfg, dict) and cfg.get("names") is not None:
                nm = cfg["names"]
                nm = [nm[k] for k in sorted(nm)] if isinstance(nm, dict) else list(nm)
                kp = cfg.get("kpt_shape")
                return [str(x) for x in nm], (tuple(int(v) for v in kp) if kp and len(kp) == 2 else None), f
    return None, None, None


def _names_from_txt(ctx, near: Path | None):
    best, best_score = None, -1
    for f in ctx.files(".txt", ".names"):
        if f.name.lower() in NAMES_FILES:
            try:
                sc = len(os.path.commonpath([str(f.parent), str(near)])) if near else 0
            except ValueError:
                sc = 0
            if sc > best_score:
                best, best_score = f, sc
    if best is None:
        return None
    lines = [ln.strip() for ln in ctx.read_text(best).splitlines() if ln.strip()]
    return lines or None


def sniff(ctx):
    files = _candidates(ctx)
    if not files:
        return None
    _names, kpt, _y = _read_cfg(ctx)
    step = max(1, len(files) // 40)
    sample = files[::step][:40]
    good = lines = 0
    kinds: dict[str, int] = {}
    for f in sample:
        try:
            txt = ctx.read_text(f, limit=200_000)
        except OSError:
            continue
        ok_file, nonempty = True, 0
        file_kinds = []
        for ln in txt.splitlines():
            t = ln.split()
            if not t:
                continue
            nonempty += 1
            if not re.fullmatch(r"-?\d+(\.0+)?", t[0]):
                ok_file = False
                break
            vals = _numbers(t[1:])
            if vals is not None and kpt and len(vals) == 4 + kpt[0] * kpt[1] and kpt[1] == 3:
                chk = vals[:4] + [v for i, v in enumerate(vals[4:]) if i % 3 != 2]          # видимость точки (0/1/2) не проверяем
            else:
                chk = vals[: 4 if (vals is not None and len(vals) == 5) else len(vals or [])] if vals is not None else []
            if vals is None or any(v < -0.01 or v > 1.01 for v in chk):
                ok_file = False
                break
            k = _line_kind(vals, kpt)
            if k is None:
                ok_file = False
                break
            file_kinds.append(k)
        if nonempty == 0:
            continue                                     # пустой файл = картинка без объектов, не голосует
        lines += 1
        if ok_file:
            good += 1
            for k in file_kinds:
                kinds[k] = kinds.get(k, 0) + 1
    if lines == 0:                                       # все файлы пустые: тоже YOLO, но уверенность низкая
        return Detection(NAME, 0.2, files, f"{len(files)} пустых txt") if files and _names else None
    if good < 0.9 * lines:
        return None
    kind_txt = ", ".join(sorted(kinds)) or "рамки"
    return Detection(NAME, 0.92 if good == lines else 0.8, files, f"{len(files)} файлов txt ({kind_txt})", {"kinds": kinds})


def _ref_for(txt: Path) -> str:
    """Путь к txt без расширения, где последняя папка labels заменена на images (для подбора картинки по хвосту пути)."""
    parts = list(txt.with_suffix("").parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() in ("labels", "label", "annotations"):
            parts[i] = "images"
            break
    return "/".join(parts[-5:])


def _quads_are_rects(files, ctx, sample: int = 60) -> bool:
    """Все 8-числовые строки выборки - параллелограммы? Прямоугольник остаётся параллелограммом при любом масштабе осей
    (координаты нормированы, размер кадра неизвестен), а произвольный четырёхугольник сегментации - нет."""
    checked = good = 0
    for f in files[:sample]:
        try:
            lines = ctx.read_text(f, limit=100_000).splitlines()
        except OSError:
            continue
        for ln in lines:
            t = ln.split()
            v = _numbers(t[1:]) if len(t) == 9 else None
            if not v:
                continue
            p = [(v[i], v[i + 1]) for i in range(0, 8, 2)]
            checked += 1
            side = max(math.hypot(p[i][0] - p[(i + 1) % 4][0], p[i][1] - p[(i + 1) % 4][1]) for i in range(4))
            dx, dy = p[0][0] + p[2][0] - p[1][0] - p[3][0], p[0][1] + p[2][1] - p[1][1] - p[3][1]
            if math.hypot(dx, dy) <= 0.12 * side + 2e-4:                  # почти параллелограмм (метки DOTA не идеальны)
                good += 1
    return checked > 0 and good >= 0.8 * checked            # часть углов может быть обрезана краем кадра


def parse(ctx, det, opt) -> Parsed:
    names, kpt, ypath = _read_cfg(ctx)
    files = det.files
    if names is None:
        names = _names_from_txt(ctx, files[0].parent if files else None)
    kinds = (det.params or {}).get("kinds", {})
    obb_mode = opt.params.get("obb") if opt and opt.params else None
    if obb_mode is None:
        obb_mode = bool(kinds.get("obb")) and not kinds.get("poly") and not kinds.get("box")
        if obb_mode and not _quads_are_rects(files, ctx):             # четырёхугольники произвольной формы - это сегментация, не OBB
            obb_mode = False
    pd = Parsed("yolo", class_names=names)
    if kpt:
        pd.keypoints = [f"p{i + 1}" for i in range(kpt[0])]
    bad = 0
    for f in files:
        try:
            txt = ctx.read_text(f)
        except OSError:
            bad += 1
            continue
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        valid = 0
        for ln in lines:
            t = ln.split()
            v = _numbers(t[1:]) if len(t) >= 5 and re.fullmatch(r"-?\d+(\.0+)?", t[0]) else None
            valid += v is not None and _line_kind(v, kpt) is not None
        if lines and valid < 0.5 * len(lines):                 # не метки (лог обучения, заметки): пропускаем весь файл
            bad += 1
            continue
        rec = Record(ref=_ref_for(f))
        rec.extra_refs = [f.stem]
        for ln in txt.splitlines():
            t = ln.split()
            if len(t) < 5:
                continue
            try:
                cls = int(float(t[0]))
            except ValueError:
                continue
            vals = _numbers(t[1:])
            if vals is None:
                continue
            k = _line_kind(vals, kpt)
            if k in ("box", "box_conf"):
                cx, cy, w, h = vals[:4]
                rec.objs.append(Obj("box", cls, [[cx - w / 2, cy - h / 2], [cx + w / 2, cy + h / 2]], norm=True,
                                    conf=vals[4] if k == "box_conf" else None))
            elif k == "pose":
                cx, cy, w, h = vals[:4]
                n_k, dim = kpt
                kps = []
                for i in range(n_k):
                    base = 4 + i * dim
                    kps.append([vals[base], vals[base + 1], int(vals[base + 2]) if dim == 3 else 2])
                rec.objs.append(Obj("pose", cls, [[cx - w / 2, cy - h / 2], [cx + w / 2, cy + h / 2]], kps=kps, norm=True))
            elif k in ("obb", "poly"):
                pts = [[vals[i], vals[i + 1]] for i in range(0, len(vals), 2)]
                rec.objs.append(Obj("obb" if (k == "obb" and obb_mode) else "poly", cls, pts, norm=True))
        pd.records.append(rec)
    if bad:
        pd.notes.append(f"Пропущено файлов, не похожих на метки: {bad}")
    if names is None:
        pd.notes.append("Названия классов не найдены (data.yaml, classes.txt, obj.names): классы будут названы class_0, class_1... "
                        "Их можно переименовать в окне импорта.")
    return pd
