"""Виртуальный список изображений: быстрая работа на десятках тысяч файлов (строки создаются только для видимых)."""
from __future__ import annotations

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt

FILTERS = ["Все изображения", "Без разметки", "С разметкой", "С подсказками", "Не проверенные"]


class ImageListModel(QAbstractListModel):
    def __init__(self, project_getter, parent=None):
        super().__init__(parent)
        self._project = project_getter
        self._rows: list[int] = []            # номера изображений в project.images, прошедших фильтр
        self._pos: dict[int, int] = {}

    # ---- построение
    def rebuild(self, mode: int, keep: int = -1):
        images = self._project().images
        rows = []
        if mode == 0:
            rows = list(range(len(images)))
        else:
            for i, e in enumerate(images):
                n_all = len(e.anns)
                n_sug = sum(1 for a in e.anns if a.suggested)
                real = n_all - n_sug
                ok = ((mode == 1 and not real) or (mode == 2 and real) or (mode == 3 and n_sug) or
                      (mode == 4 and not e.reviewed))
                if ok or i == keep:
                    rows.append(i)
        self.beginResetModel()
        self._rows = rows
        self._pos = {i: r for r, i in enumerate(rows)}
        self.endResetModel()

    def image_index(self, row: int) -> int:
        return self._rows[row] if 0 <= row < len(self._rows) else -1

    def row_of(self, image_idx: int) -> int:
        return self._pos.get(image_idx, -1)

    def touch(self, image_idx: int):
        r = self._pos.get(image_idx)
        if r is not None:
            ix = self.index(r)
            self.dataChanged.emit(ix, ix)

    # ---- QAbstractListModel
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        i = self._rows[index.row()]
        if role == Qt.DisplayRole:
            return self.text(self._project().images[i])
        if role == Qt.UserRole:
            return i
        if role == Qt.ToolTipRole:
            return self._project().images[i].path
        return None

    @staticmethod
    def text(e) -> str:
        sug = sum(1 for a in e.anns if a.suggested)
        real = len(e.anns) - sug
        mark = "✔" if e.reviewed else ("●" if real else ("◐" if sug else "○"))
        return f"{mark} {e.name}   [{real}{'+' + str(sug) + '?' if sug else ''}]"
