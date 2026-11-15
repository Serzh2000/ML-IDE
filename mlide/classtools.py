"""Bulk class operations on a Project (no Qt)."""
from __future__ import annotations

from .model import Project


def class_usage(project: Project) -> list:
    """Number of confirmed (non-suggested) annotations per class index."""
    counts = [0] * len(project.classes)
    for im in project.images:
        for a in im.anns:
            if not a.suggested and 0 <= a.cls < len(counts):
                counts[a.cls] += 1
    return counts


def remap_classes(project: Project, mapping: dict) -> None:
    """mapping: old index -> new index (in the *old* numbering space of the surviving classes),
    or None to delete those objects. Classes not listed keep themselves. Afterwards every class that
    no object-mapping targets and that is itself remapped/deleted is removed from the class list,
    and the remaining classes are renumbered keeping their order."""
    n = len(project.classes)
    full = {i: mapping.get(i, i) for i in range(n)}
    _dropped = {i for i in range(n) if full[i] != i}
    if any(t is not None and t in _dropped for t in full.values()):
        raise ValueError("Целевой класс сам удаляется/переносится (цепочка или обмен не поддерживается)")
    for im in project.images:
        new = []
        for a in im.anns:
            t = full.get(a.cls, a.cls)
            if t is None:
                continue
            a.cls = t
            new.append(a)
        im.anns = new
    # classes that were mapped away (src != dst) are dropped
    dropped = {i for i in range(n) if full[i] != i}
    keep = [i for i in range(n) if i not in dropped]
    renum = {old: new for new, old in enumerate(keep)}
    for im in project.images:
        for a in im.anns:
            a.cls = renum[a.cls]
    project.classes = [project.classes[i] for i in keep]
    project.dirty = True


def merge_classes(project: Project, src: int, dst: int) -> None:
    if src == dst:
        return
    remap_classes(project, {src: dst})


def rename_class(project: Project, idx: int, new_name: str) -> None:
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("Пустое имя класса")
    for i, c in enumerate(project.classes):
        if i != idx and c["name"] == new_name:
            merge_classes(project, idx, i)
            return
    project.classes[idx]["name"] = new_name
    project.dirty = True


def move_class(project: Project, idx: int, new_pos: int) -> None:
    n = len(project.classes)
    new_pos = max(0, min(n - 1, new_pos))
    if idx == new_pos:
        return
    order = list(range(n))
    order.insert(new_pos, order.pop(idx))        # order[new_index] = old_index
    renum = {old: new for new, old in enumerate(order)}
    project.classes = [project.classes[o] for o in order]
    for im in project.images:
        for a in im.anns:
            a.cls = renum[a.cls]
    project.dirty = True
