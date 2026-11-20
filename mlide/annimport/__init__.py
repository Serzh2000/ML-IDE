"""Универсальный импорт разметки: сам узнаёт формат (YOLO, COCO, VOC, LabelMe, CVAT, VIA, Label Studio, CSV/JSON-таблицы...),
находит картинки по ссылкам из файлов разметки и записывает всё в проект.

Использование:
    ctx = SourceContext([папка_или_файлы], images_dir=None)
    dets = detect(ctx)                       # [Detection] - найденные форматы по убыванию уверенности
    parsed = parse(ctx, dets[0], Options())  # Parsed
    pv = preview(ctx, parsed, Options())     # сколько картинок найдено, объектов, классы - до записи в проект
    rep = run_import(project, ctx, dets[0], Options())
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from .apply import apply_parsed, class_keys, name_of  # noqa: F401
from .context import SourceContext  # noqa: F401
from .resolve import ImageResolver  # noqa: F401
from .types import ApplyReport, Detection, Obj, Options, Parsed, Record  # noqa: F401

# порядок = порядок показа при равной уверенности; модули, которых нет, пропускаются
PARSER_NAMES = ["coco", "voc", "labelme", "cvat", "via", "labelstudio", "createml", "txtper", "yolo", "pairs", "table", "folders"]
_loaded: dict | None = None


def get_parsers() -> dict:
    global _loaded
    if _loaded is None:
        _loaded = {}
        for n in PARSER_NAMES:
            try:
                m = importlib.import_module(f"{__name__}.parsers.{n}")
            except ModuleNotFoundError as e:
                if e.name and e.name.endswith(f"parsers.{n}"):
                    continue
                raise
            _loaded[m.NAME] = m
    return _loaded


def detect(ctx: SourceContext) -> list[Detection]:
    out = []
    for m in get_parsers().values():
        try:
            d = m.sniff(ctx)
        except Exception:  # noqa  (один сломанный разборщик не должен ломать остальные)
            d = None
        if d and d.score > 0:
            out.append(d)
    order = {n: i for i, n in enumerate(PARSER_NAMES)}
    return sorted(out, key=lambda d: (-d.score, order.get(d.parser, 99)))


def title_of(parser: str, lang: str = "ru") -> str:
    m = get_parsers().get(parser)
    return (getattr(m, "TITLE", {}) or {}).get(lang) or parser


def parse(ctx: SourceContext, det: Detection, opt: Options | None = None) -> Parsed:
    return get_parsers()[det.parser].parse(ctx, det, opt or Options())


def build_resolver(ctx: SourceContext, project=None, widen: bool = False) -> ImageResolver:
    res = ImageResolver(ctx.image_roots(), project)
    if widen:
        for r in ctx.parent_roots():
            res.add_root(r, max_files=100_000)
    return res


@dataclass
class Preview:
    records: int = 0
    objects: int = 0
    matched: int = 0
    unmatched: int = 0
    ambiguous: int = 0
    classes: list = field(default_factory=list)
    unmatched_samples: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    widened: bool = False


def preview(ctx: SourceContext, parsed: Parsed, project=None) -> Preview:
    """Считает, сколько записей найдёт свои картинки, - ничего не меняя в проекте."""
    pv = Preview(records=len(parsed.records), classes=class_keys(parsed), notes=list(parsed.notes))
    pv.objects = sum(len(r.objs) for r in parsed.records)
    for widen in (False, True):
        res = build_resolver(ctx, project, widen)
        pv.matched = pv.unmatched = pv.ambiguous = 0
        pv.unmatched_samples = []
        for r in parsed.records:
            p, st = res.find(r.ref, r.extra_refs)
            if p is None:
                pv.unmatched += 1
                if len(pv.unmatched_samples) < 8:
                    pv.unmatched_samples.append(r.ref)
            else:
                pv.matched += 1
                pv.ambiguous += st == "ambiguous"
        pv.widened = widen
        if pv.records == 0 or pv.matched >= 0.5 * pv.records or not ctx.parent_roots():
            break
    return pv


def run_import(project, ctx: SourceContext, det: Detection, opt: Options | None = None, progress=None, parsed: Parsed | None = None) -> ApplyReport:
    opt = opt or Options()
    parsed = parsed or parse(ctx, det, opt)
    pv = preview(ctx, parsed, project)
    res = build_resolver(ctx, project, widen=pv.widened)
    return apply_parsed(project, parsed, res, opt, progress)
