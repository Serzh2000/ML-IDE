"""Общие типы универсального импорта разметки (без Qt).

Любой формат разбирается в одну промежуточную форму: Parsed -> [Record -> [Obj]]. Дальше общий код сам находит картинки
(resolve.py), переводит координаты, сопоставляет классы и записывает всё в проект (apply.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Obj:
    """Один размеченный объект.

    kind: "box" - pts=[[x1,y1],[x2,y2]]; "poly" - pts=[[x,y],...]; "obb" - 4 угла; "pose" - рамка как box + kps=[[x,y,v],...].
    cls: имя класса (str) ИЛИ числовой номер (int), который расшифровывается через Parsed.class_names.
    norm: координаты в долях кадра (0..1), а не в пикселях.
    """
    kind: str
    cls: object
    pts: list
    kps: list | None = None
    norm: bool = False
    conf: float | None = None


@dataclass
class Record:
    """Описание одной картинки: ссылка на файл (как записана в источнике), размер (если известен), объекты, класс кадра."""
    ref: str
    objs: list = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    tag: object = None            # класс всего кадра (классификация): имя или номер
    extra_refs: list = field(default_factory=list)   # запасные варианты ссылки (id, путь из другого поля и т.п.)


@dataclass
class Parsed:
    format: str
    records: list = field(default_factory=list)
    class_names: list | None = None          # расшифровка числовых классов: индекс -> имя
    keypoints: list | None = None            # имена ключевых точек (pose)
    skeleton: list | None = None
    notes: list = field(default_factory=list)      # предупреждения разбора (показываются пользователю)


@dataclass
class Detection:
    """Результат «узнавания» формата: что за формат, насколько уверены и какие файлы он будет читать."""
    parser: str
    score: float                              # 0..1
    files: list = field(default_factory=list)
    summary: str = ""                         # одна строка для окна: «1250 файлов .xml»
    params: dict = field(default_factory=dict)  # настройки, найденные при разборе (например, сопоставление столбцов таблицы)


@dataclass
class Options:
    """Настройки применения разобранной разметки к проекту."""
    existing: str = "append"          # что делать с картинками, которые уже в проекте и уже имеют разметку: skip | replace | append
    add_missing_images: bool = True   # добавлять в проект найденные картинки, которых ещё нет
    import_tags: bool = True          # переносить класс кадра (классификация)
    images_dir: str | None = None     # дополнительная папка с картинками
    class_map: dict = field(default_factory=dict)   # имя в источнике -> имя класса проекта; None - пропустить объекты этого класса
    params: dict = field(default_factory=dict)      # параметры парсера (сопоставление столбцов и т.п.)


@dataclass
class ApplyReport:
    images_added: int = 0
    images_updated: int = 0           # картинка уже была в проекте, разметка добавлена/заменена
    images_skipped: int = 0
    objects_added: int = 0
    tags_set: int = 0
    unmatched: list = field(default_factory=list)    # ссылки, для которых не нашлось файла
    ambiguous: list = field(default_factory=list)    # ссылки с несколькими подходящими файлами
    classes_added: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        s = f"Картинок добавлено: {self.images_added}, обновлено: {self.images_updated}, объектов: {self.objects_added}"
        if self.tags_set:
            s += f", классов кадров: {self.tags_set}"
        if self.classes_added:
            s += f", новых классов: {len(self.classes_added)}"
        if self.images_skipped:
            s += f". Пропущено (уже размечены): {self.images_skipped}"
        if self.unmatched:
            s += f". Картинки не найдены: {len(self.unmatched)}"
        if self.ambiguous:
            s += f". Неоднозначных совпадений: {len(self.ambiguous)}"
        return s + "."
