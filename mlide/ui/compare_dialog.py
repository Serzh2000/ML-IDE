"""Сравнение запусков обучения: таблица метрик, наложенные кривые и вывод простым языком."""
from __future__ import annotations

from . import winflags

import csv
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget,
                               QTableWidgetItem, QTextBrowser, QVBoxLayout)

from .. import knowledge as K
from .learn import LineChart, cards_html

COLORS = ["#4da3ff", "#ff9f43", "#2ecc71", "#fd79a8", "#a29bfe", "#feca57", "#00d2d3", "#ff6b6b"]


@dataclass
class RunInfo:
    name: str
    path: Path
    S: dict = field(default_factory=dict)
    args: dict = field(default_factory=dict)
    best_i: int = 0
    size_mb: float = 0.0

    def val(self, key, i=None):
        v = self.S.get(key) or []
        i = self.best_i if i is None else i
        return v[i] if 0 <= i < len(v) and v[i] is not None else None


def scan_runs(folder: Path) -> list[RunInfo]:
    runs = []
    for csvp in sorted(folder.rglob("results.csv"), key=lambda p: p.stat().st_mtime):
        try:
            with open(csvp, encoding="utf-8", newline="") as f:
                S = K.parse_rows(list(csv.DictReader(f)))
        except Exception:
            continue
        if not S["epoch"]:
            continue
        args = {}
        try:
            args = yaml.safe_load((csvp.parent / "args.yaml").read_text(encoding="utf-8")) or {}
        except Exception:
            pass
        fit = [0.1 * (S["map50"][i] or 0) + 0.9 * (S["map"][i] or 0) for i in range(len(S["epoch"]))]
        bi = max(range(len(fit)), key=lambda i: fit[i])
        size = 0.0
        for n in ("best.onnx", "best.pt"):
            w = csvp.parent / "weights" / n
            if w.exists():
                size = w.stat().st_size / 1048576
                break
        try:
            label = str(csvp.parent.relative_to(folder))
        except ValueError:
            label = str(csvp.parent)
        runs.append(RunInfo(f"{folder.name}/{label}", csvp.parent, S, args, bi, size))
    return runs


class CompareDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сравнение запусков обучения")
        self.resize(1200, 780)
        self.project = project
        self.runs: list[RunInfo] = []
        lay = winflags.scroll_layout(self)
        row = QHBoxLayout()
        self.folder = QLineEdit(project.export.get("out_dir", ""))
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        add = QPushButton("Добавить ещё папку…")
        add.clicked.connect(self._add)
        row.addWidget(QLabel("Папка с запусками:"))
        row.addWidget(self.folder, 1)
        row.addWidget(b)
        row.addWidget(add)
        lay.addLayout(row)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Запуск", "Модель", "imgsz", "Эпох", "mAP50", "mAP50-95", "Precision", "Recall", "Файл, МБ"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 300)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(230)
        self.table.itemSelectionChanged.connect(self._selection)
        lay.addWidget(self.table)
        lay.addWidget(QLabel("<span style='color:#8b92a1'>Выделите несколько строк (Ctrl/Shift), чтобы наложить кривые и получить вывод.</span>"))
        row = QHBoxLayout()
        self.chart_box = QVBoxLayout()
        self.chart = LineChart("mAP50 по эпохам", "", [])
        self.chart_box.addWidget(self.chart)
        row.addLayout(self.chart_box, 3)
        self.verdict = QTextBrowser()
        row.addWidget(self.verdict, 2)
        lay.addLayout(row, 1)
        self._scan()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Папка с запусками", self.folder.text())
        if d:
            self.folder.setText(d)
            self.runs = []
            self._scan()

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Ещё одна папка с запусками")
        if d:
            self.folder.setText(self.folder.text() + (";" if self.folder.text() else "") + d)
            self._scan()

    def _scan(self):
        self.runs = []
        for part in [p.strip() for p in self.folder.text().split(";") if p.strip()]:
            if Path(part).exists():
                self.runs += scan_runs(Path(part))
        self.table.setRowCount(0)
        f = lambda v: "-" if v is None else f"{v:.3f}"
        for r in self.runs:
            i = self.table.rowCount()
            self.table.insertRow(i)
            vals = [r.name, str(r.args.get("model", "?")), str(r.args.get("imgsz", "?")), str(len(r.S["epoch"])),
                    f(r.val("map50")), f(r.val("map")), f(r.val("p")), f(r.val("r")), f"{r.size_mb:.1f}" if r.size_mb else "-"]
            for c, t in enumerate(vals):
                self.table.setItem(i, c, QTableWidgetItem(t))
        if not self.runs:
            self.verdict.setHtml("<p>Запуски не найдены. Укажите папку, где есть <code>runs/…/results.csv</code>.</p>")
        elif len(self.runs) >= 2:
            self.table.selectAll()

    def _selection(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        sel = [self.runs[r] for r in rows if r < len(self.runs)]
        series = [(f"r{k}", Path(r.name).name if r.name.endswith("train") else r.name.split("/", 1)[-1], COLORS[k % len(COLORS)])
                  for k, r in enumerate(sel)]
        new = LineChart("mAP50 по эпохам - чем выше, тем лучше", "", series)
        longest = max((r.S["epoch"] for r in sel), key=len, default=[])
        new.set_data(longest, {f"r{k}": r.S["map50"] for k, r in enumerate(sel)})
        self.chart_box.replaceWidget(self.chart, new)
        self.chart.deleteLater()
        self.chart = new
        self.verdict.setHtml(cards_html(self._verdict(sel)))

    def _verdict(self, sel):
        if not sel:
            return [("info", "Выберите запуски", "Кликните по строкам таблицы.")]
        if len(sel) == 1:
            r = sel[0]
            return [("info", "Один запуск", f"Лучшая эпоха {int(r.val('epoch'))}: mAP50 = {r.val('map50'):.3f}, mAP50-95 = {r.val('map'):.3f}.")]
        by = sorted(sel, key=lambda r: -(r.val("map") or 0))
        best, second = by[0], by[1]
        d50 = (best.val("map50") or 0) - (second.val("map50") or 0)
        d95 = (best.val("map") or 0) - (second.val("map") or 0)
        out = [("ok", f"Лучший: {best.name}",
                f"mAP50-95 = {best.val('map'):.3f}, mAP50 = {best.val('map50'):.3f}. Отрыв от следующего: "
                f"{d95:+.3f} по mAP50-95, {d50:+.3f} по mAP50.")]
        if abs(d95) < 0.02:
            out.append(("warn", "Разница в пределах шума",
                        "Меньше 0.02: на небольшой проверочной выборке такой разброс случаен. Не делайте выводов, пока нет "
                        "хотя бы нескольких сотен проверочных объектов или отдельного test."))
        # факторы различий
        diffs = []
        for k in ("model", "imgsz", "epochs", "freeze", "lr0", "optimizer", "batch"):
            vals = {str(r.args.get(k)) for r in sel}
            if len(vals) > 1:
                diffs.append(f"<b>{k}</b>: " + ", ".join(f"{Path(r.name).name}={r.args.get(k)}" for r in sel))
        if diffs:
            out.append(("info", "Чем запуски отличались", "<br>".join(diffs)))
        sizes = [r for r in sel if r.size_mb]
        if len(sizes) >= 2:
            small = min(sizes, key=lambda r: r.size_mb)
            if small is not best and (best.val("map") or 0) - (small.val("map") or 0) < 0.03:
                out.append(("info", "Компромисс размера и качества",
                            f"{small.name} (файл меньше ×{best.size_mb / max(small.size_mb, 0.1):.1f}) почти не уступает по качеству - "
                            "для слабого устройства, возможно, лучший выбор."))
        return out
