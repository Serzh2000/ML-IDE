"""Настройка ключевых точек для задачи pose: названия, скелет (какие точки соединять) и пары «левая-правая»."""
from __future__ import annotations

import re

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QMessageBox, QPlainTextEdit, QVBoxLayout)

from .. import keypoints as KP
from .dialogs import add_tips

PRESETS = {
    "Человек (COCO, 17 точек)": ("nose\nleft_eye\nright_eye\nleft_ear\nright_ear\nleft_shoulder\nright_shoulder\nleft_elbow\nright_elbow\n"
                                 "left_wrist\nright_wrist\nleft_hip\nright_hip\nleft_knee\nright_knee\nleft_ankle\nright_ankle",
                                 "left_shoulder-right_shoulder\nleft_shoulder-left_elbow\nleft_elbow-left_wrist\nright_shoulder-right_elbow\n"
                                 "right_elbow-right_wrist\nleft_shoulder-left_hip\nright_shoulder-right_hip\nleft_hip-right_hip\n"
                                 "left_hip-left_knee\nleft_knee-left_ankle\nright_hip-right_knee\nright_knee-right_ankle\nnose-left_eye\nnose-right_eye"),
}


def parse_names(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_pairs(text: str, names: list[str]) -> tuple[list[list[int]], list[str]]:
    """Строки вида «a-b»: имена точек или их номера (с 1). Возвращает (пары индексов, ошибки)."""
    idx = {n: i for i, n in enumerate(names)}
    pairs, errs = [], []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [x.strip() for x in re.split(r"\s*[-–,;]\s*", ln) if x.strip()]
        if len(parts) != 2:
            errs.append(f"«{ln}»: нужно две точки через дефис")
            continue
        ij = []
        for x in parts:
            if x in idx:
                ij.append(idx[x])
            elif x.isdigit() and 1 <= int(x) <= len(names):
                ij.append(int(x) - 1)
            else:
                errs.append(f"«{ln}»: неизвестная точка «{x}»")
        if len(ij) == 2 and ij[0] != ij[1]:
            pairs.append(ij)
    return pairs, errs


class KeypointsDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ключевые точки")
        self.resize(640, 640)
        self.project = project
        lay = QVBoxLayout(self)
        intro = QLabel("Ключевые точки (pose): суставы человека, углы объекта, глаза - любые точки в <b>одном и том же порядке</b> "
                       "на каждом объекте. Задайте их один раз для проекта: потом инструмент «Ключевые точки» будет просить ставить "
                       "их по очереди.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(QLabel("<b>Названия точек</b> (по одной в строке, порядок важен)"))
        self.names = QPlainTextEdit("\n".join(project.keypoints))
        lay.addWidget(self.names, 3)
        lay.addWidget(QLabel("<b>Скелет</b> - какие точки соединять линиями (по паре в строке: <code>nose-left_eye</code> или <code>1-2</code>)"))
        self.skeleton = QPlainTextEdit("\n".join(f"{project.keypoints[i]}-{project.keypoints[j]}" for i, j in project.skeleton
                                                 if i < len(project.keypoints) and j < len(project.keypoints)))
        lay.addWidget(self.skeleton, 2)
        lay.addWidget(QLabel("<b>Пары «левая-правая»</b> (нужны, чтобы отражение по горизонтали не путало стороны)"))
        self.flips = QPlainTextEdit("\n".join(f"{project.keypoints[i]}-{project.keypoints[j]}" for i, j in project.flip_pairs
                                              if i < len(project.keypoints) and j < len(project.keypoints)))
        lay.addWidget(self.flips, 2)
        row = QDialogButtonBox()
        pre = row.addButton("Подставить: человек (17 точек)", QDialogButtonBox.ActionRole)
        pre.clicked.connect(self._preset)
        row.addButton(QDialogButtonBox.Ok)
        row.addButton(QDialogButtonBox.Cancel)
        row.accepted.connect(self._apply)
        row.rejected.connect(self.reject)
        lay.addWidget(row)
        add_tips(self, {
            self.names: "Каждая строка - одна точка. Порядок нельзя менять после разметки (иначе точки перепутаются).",
            self.skeleton: "Линии между точками рисуются при разметке и помогают проверять результат. Для обучения не нужны.",
            self.flips: "Например left_eye-right_eye. Если оставить пустым, при обучении отражение по горизонтали будет отключено.",
        })

    def _preset(self):
        self.names.setPlainText(KP.names_text(KP.COCO_NAMES))
        self.skeleton.setPlainText(KP.pairs_text(KP.COCO_SKELETON, KP.COCO_NAMES))
        self.flips.setPlainText(KP.pairs_text(KP.COCO_FLIP, KP.COCO_NAMES))

    def _apply(self):
        names = parse_names(self.names.toPlainText())
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Ключевые точки", "Названия точек должны быть разными.")
            return
        sk, e1 = parse_pairs(self.skeleton.toPlainText(), names)
        fl, e2 = parse_pairs(self.flips.toPlainText(), names)
        if e1 or e2:
            QMessageBox.warning(self, "Ключевые точки", "\n".join((e1 + e2)[:8]))
            return
        old = self.project.keypoints
        used = sum(1 for e in self.project.images for a in e.anns if a.kind == "pose")
        if used and names != old:
            if names[:len(old)] != old and old:
                if QMessageBox.question(self, "Ключевые точки", f"В проекте уже есть {used} поз. Вы изменили или переставили существующие точки: "
                                        "размеченные точки могут не совпасть с новыми названиями. Продолжить?") != QMessageBox.Yes:
                    return
            for e in self.project.images:                                  # выравниваем число точек у готовых поз
                for a in e.anns:
                    if a.kind == "pose" and a.kps is not None:
                        a.kps = [list(k) for k in a.kps][:len(names)] + [[0.0, 0.0, 0]] * max(0, len(names) - len(a.kps))
        self.project.keypoints, self.project.skeleton, self.project.flip_pairs = names, sk, fl
        self.project.dirty = True
        self.accept()
