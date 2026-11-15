"""Просмотр модели: схема графа ONNX, список слоёв, сводка и пояснения простым языком."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QDialog, QFileDialog, QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import onnxparse as O

COLORS = {"conv": "#3d6fd9", "act": "#2e9e6b", "pool": "#d98a2b", "shape": "#6b7280", "math": "#9a5bd6", "other": "#4b5563",
          "io": "#d94f6b"}
NODE_W, NODE_H, GAP_X, GAP_Y = 168, 38, 26, 34


class NodeItem(QGraphicsRectItem):
    def __init__(self, key, title, sub, color, w=NODE_W, h=NODE_H):
        super().__init__(0, 0, w, h)
        self.key = key
        self.base_color = QColor(color)
        self.setBrush(QBrush(self.base_color))
        self.setPen(QPen(self.base_color.lighter(140), 1.2))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(2)
        t = QGraphicsSimpleTextItem(title, self)
        t.setBrush(QBrush(QColor("white")))
        f = QFont("Segoe UI", 9)
        f.setBold(True)
        t.setFont(f)
        t.setPos(8, 3)
        s = QGraphicsSimpleTextItem(sub[:26], self)
        s.setBrush(QBrush(QColor(255, 255, 255, 190)))
        s.setFont(QFont("Segoe UI", 8))
        s.setPos(8, 19)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        sel = self.isSelected()
        p.setPen(QPen(QColor("white") if sel else self.pen().color(), 2.5 if sel else 1.2))
        p.setBrush(self.brush())
        p.drawRoundedRect(self.rect(), 7, 7)


class GraphView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#14161a")))
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        s = self.transform().m11() * f
        if 0.03 <= s <= 4:
            self.scale(f, f)


class ModelViewerDialog(QDialog):
    def __init__(self, parent=None, path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Просмотр модели")
        self.resize(1400, 860)
        self.model: O.OnnxModel | None = None
        self.items: dict[object, NodeItem] = {}
        self.edges: list[tuple[object, object, QGraphicsPathItem]] = []
        self._matches: list = []
        self._mi = 0

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.path = QLineEdit(path)
        self.path.setPlaceholderText("Файл модели .onnx")
        self.path.setToolTip("<p>Откройте .onnx: обученную модель можно получить в «Экспорт модели → ONNX». "
                             "Файл .pt напрямую показать нельзя - сначала экспортируйте его в ONNX.</p>")
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        load = QPushButton("Показать")
        load.setProperty("primary", True)
        load.clicked.connect(lambda: self.load(self.path.text().strip()))
        top.addWidget(QLabel("Модель:"))
        top.addWidget(self.path, 1)
        top.addWidget(b)
        top.addWidget(load)
        lay.addLayout(top)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # --- схема
        scheme = QWidget()
        sv = QVBoxLayout(scheme)
        sv.setContentsMargins(0, 6, 0, 0)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск слоя: Conv, Concat, model.10…  (Enter - следующий)")
        self.search.returnPressed.connect(self._find)
        self.search.setToolTip("<p>Ищет по типу и имени слоя; Enter переходит к следующему совпадению.</p>")
        fit = QPushButton("Вписать")
        fit.clicked.connect(self._fit)
        bar.addWidget(self.search, 1)
        bar.addWidget(fit)
        legend = QLabel("  ".join(f"<span style='color:{COLORS[k]}'>■</span> {t}" for k, t in (
            ("conv", "свёртки"), ("act", "активации"), ("pool", "пулинг/масштаб"), ("shape", "формы/склейка"),
            ("math", "арифметика"), ("io", "вход/выход"))))
        legend.setStyleSheet("color:#8b92a1")
        bar.addWidget(legend)
        sv.addLayout(bar)
        sp = QSplitter(Qt.Horizontal)
        self.scene = QGraphicsScene(self)
        self.view = GraphView()
        self.view.setScene(self.scene)
        self.scene.selectionChanged.connect(self._on_select)
        sp.addWidget(self.view)
        self.detail = QTextBrowser()
        self.detail.setMinimumWidth(320)
        sp.addWidget(self.detail)
        sp.setSizes([980, 380])
        sv.addWidget(sp, 1)
        self.tabs.addTab(scheme, "Схема")

        # --- слои
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["№", "Тип", "Имя", "Параметры слоя", "Веса", "Параметров"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._row_open)
        self.tabs.addTab(self.table, "Слои")

        # --- сводка
        self.summary = QTextBrowser()
        self.tabs.insertTab(0, self.summary, "Сводка")
        self.tabs.setCurrentIndex(0)
        if path and Path(path).exists():
            self.load(path)
        else:
            self.summary.setHtml("<h3>Выберите файл .onnx</h3><p>Здесь появятся сводка по модели, схема слоёв и список "
                                 "слоёв с пояснениями. Клик по слою на схеме показывает, что он делает.</p>")

    # ------------------------------------------------------------------ загрузка
    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Модель ONNX", self.path.text(), "ONNX (*.onnx)")
        if p:
            self.path.setText(p)
            self.load(p)

    def load(self, path: str):
        if not path or not Path(path).exists():
            QMessageBox.information(self, "Просмотр модели", "Укажите существующий файл .onnx.")
            return
        if path.lower().endswith(".pt"):
            QMessageBox.information(self, "Просмотр модели", "Файл .pt показать нельзя: сначала экспортируйте его в ONNX "
                                    "(Инструменты → Экспорт модели).")
            return
        try:
            self.model = O.load_onnx(path)
        except Exception as e:  # noqa
            QMessageBox.warning(self, "Просмотр модели", f"Не удалось прочитать модель: {type(e).__name__}: {e}")
            return
        self.path.setText(path)
        self._summary()
        self._table()
        self._build_graph()

    # ------------------------------------------------------------------ сводка
    def _summary(self):
        m = self.model
        ops = Counter(n.op for n in m.nodes)
        conv = ops.get("Conv", 0)
        rows = "".join(f"<tr><td><b>{op}</b></td><td align='right'>{c}</td><td>{O.OP_DOC.get(op, ('', ''))[0]}</td></tr>"
                       for op, c in ops.most_common())
        ins = "<br>".join(f"<code>{n}</code>: {dt} {sh}" for n, dt, sh in m.inputs)
        size_mb = m.size / 1048576
        fp = "float16" if any(dt == "float16" for _d, dt in m.inits.values()) else "float32"
        hint = ""
        if m.inputs and len(m.inputs[0][2]) == 4 and all(isinstance(x, int) for x in m.inputs[0][2][2:]):
            hh, ww = m.inputs[0][2][2:]
            hint = f"Вход рассчитан на картинку <b>{ww}×{hh}</b>: при запуске её нужно привести к этому размеру."
        self.summary.setHtml(
            f"<h2>{Path(m.path).name}</h2>"
            f"<p><b>{len(m.nodes)}</b> слоёв (из них свёрток <b>{conv}</b>) · <b>{m.params / 1e6:.2f} млн</b> параметров · "
            f"{size_mb:.1f} МБ · веса {fp} · opset {m.opsets.get('ai.onnx', '?')} · создано: {m.producer or '?'}</p>"
            f"<h3>Вход</h3><p>{ins}</p>{('<p>' + hint + '</p>') if hint else ''}"
            f"<h3>Выход</h3><p>{O.describe_outputs(m)}</p>"
            "<h3>Из чего состоит модель</h3>"
            "<p>Современный детектор - это цепочка блоков «свёртка → активация → …»: <b>Conv</b> ищет признаки, "
            "<b>Sigmoid + Mul</b> вместе дают активацию SiLU, <b>Concat/Split/Add</b> смешивают потоки, "
            "<b>MaxPool/Resize</b> меняют масштаб. В конце всё сводится к списку кандидатов-рамок.</p>"
            f"<table cellpadding='4'><tr><th align='left'>Операция</th><th>шт.</th><th align='left'>Что это</th></tr>{rows}</table>"
            "<p style='color:#8b92a1'>Совет: параметров больше - модель точнее, но медленнее и требует больше данных. "
            "Для слабых устройств берут размер n и маленький вход.</p>")

    # ------------------------------------------------------------------ таблица
    def _table(self):
        m = self.model
        self.table.setRowCount(0)
        for n in m.nodes:
            w = next(((m.inits[i][0]) for i in n.inputs[1:] if i in m.inits and len(m.inits[i][0]) > 1), None)
            if w is None:
                w = next((m.inits[i][0] for i in n.inputs if i in m.inits), None)
            params = 0
            for i in n.inputs:
                if i in m.inits:
                    p = 1
                    for d in m.inits[i][0]:
                        p *= max(d, 1)
                    params += p
            attrs = ", ".join(f"{k}={v}" for k, v in n.attrs.items() if k in ("kernel_shape", "strides", "group", "axis", "mode"))
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, txt in enumerate((str(n.idx), n.op, n.name, attrs, str(w) if w else "", f"{params:,}".replace(",", " ") if params else "")):
                self.table.setItem(r, c, QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(2, 320)

    def _row_open(self, row, _c):
        self.tabs.setCurrentIndex(1)
        key = self.model.nodes[row].idx
        it = self.items.get(key)
        if it:
            self.scene.clearSelection()
            it.setSelected(True)
            self.view.centerOn(it)

    # ------------------------------------------------------------------ граф
    def _build_graph(self):
        m = self.model
        self.scene.clear()
        self.items.clear()
        self.edges.clear()
        producer = {}
        level = {}
        for name, _dt, _sh in m.inputs:
            producer[name] = ("in", name)
            level[("in", name)] = 0
        order = []
        for n in m.nodes:
            lv = 0
            for i in n.inputs:
                src = producer.get(i)
                if src is not None:
                    lv = max(lv, level[src] + 1)
            level[n.idx] = lv
            for o in n.outputs:
                producer[o] = n.idx
            order.append(n.idx)
        out_nodes = []
        for name, _dt, _sh in m.outputs:
            key = ("out", name)
            src = producer.get(name)
            level[key] = (level[src] + 1) if src is not None else 1
            out_nodes.append((key, src))
        # позиции: по уровням, порядок по барицентру родителей
        levels: dict[int, list] = {}
        for k, lv in level.items():
            levels.setdefault(lv, []).append(k)
        parents: dict[object, list] = {n.idx: [producer[i] for i in n.inputs if i in producer] for n in m.nodes}
        for key, src in out_nodes:
            parents[key] = [src] if src is not None else []
        xpos: dict[object, float] = {}
        for lv in sorted(levels):
            keys = levels[lv]
            if lv == 0:
                for j, k in enumerate(keys):
                    xpos[k] = j
                continue
            des = {k: (sum(xpos[p] for p in parents.get(k, []) if p in xpos) / max(1, len([p for p in parents.get(k, []) if p in xpos])))
                   for k in keys}
            last = -1e9
            for k in sorted(keys, key=lambda k: (des[k], str(k))):
                x = max(des[k], last + 1)
                xpos[k] = x
                last = x
        # линии
        by_idx = {n.idx: n for n in m.nodes}
        pos = {k: QPointF(xpos[k] * (NODE_W + GAP_X), level[k] * (NODE_H + GAP_Y)) for k in level}
        # узлы
        for k, lv in level.items():
            if isinstance(k, tuple):
                title = "Вход" if k[0] == "in" else "Выход"
                shape = next((sh for n, _d, sh in (m.inputs if k[0] == "in" else m.outputs) if n == k[1]), [])
                it = NodeItem(k, title, f"{k[1]} {shape}", COLORS["io"])
            else:
                n = by_idx[k]
                sub = "/".join(n.name.strip("/").split("/")[:-1]) or n.name
                it = NodeItem(k, n.op, sub, COLORS[O.category(n.op)])
            it.setPos(pos[k])
            self.scene.addItem(it)
            self.items[k] = it
        for k, ps in parents.items():
            for p in ps:
                if p in self.items and k in self.items:
                    a = self.items[p]
                    b = self.items[k]
                    path = QPainterPath(QPointF(a.pos().x() + NODE_W / 2, a.pos().y() + NODE_H))
                    ex, ey = b.pos().x() + NODE_W / 2, b.pos().y()
                    my = (a.pos().y() + NODE_H + ey) / 2
                    path.cubicTo(QPointF(a.pos().x() + NODE_W / 2, my), QPointF(ex, my), QPointF(ex, ey))
                    e = QGraphicsPathItem(path)
                    e.setPen(QPen(QColor(120, 128, 142, 140), 1.2))
                    e.setZValue(1)
                    self.scene.addItem(e)
                    self.edges.append((p, k, e))
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-60, -60, 60, 60))
        self._fit()
        self.detail.setHtml("<p>Кликните по слою на схеме. Колесо - масштаб, перетаскивание - сдвиг. "
                            "Цвет показывает тип слоя (см. легенду сверху).</p>")

    def _fit(self):
        r = self.scene.itemsBoundingRect()
        if r.isEmpty():
            return
        # длинная вертикальная схема: вписываем по ширине и показываем начало
        self.view.resetTransform()
        s = max(0.05, min(1.0, self.view.viewport().width() / (r.width() + 80)))
        self.view.scale(s, s)
        self.view.centerOn(r.center().x(), r.top() + self.view.viewport().height() / (2 * s))

    def _find(self):
        q = self.search.text().strip().lower()
        if not q or not self.model:
            return
        if not self._matches or getattr(self, "_q", "") != q:
            self._q = q
            self._matches = [n.idx for n in self.model.nodes if q in n.op.lower() or q in n.name.lower()]
            self._mi = 0
        if not self._matches:
            self.detail.setHtml("<p>Ничего не найдено.</p>")
            return
        key = self._matches[self._mi % len(self._matches)]
        self._mi += 1
        it = self.items.get(key)
        if it:
            self.scene.clearSelection()
            it.setSelected(True)
            self.view.centerOn(it)

    # ------------------------------------------------------------------ детали слоя
    def _on_select(self):
        sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        for _p, _k, e in self.edges:
            e.setPen(QPen(QColor(120, 128, 142, 140), 1.2))
        if not sel or not self.model:
            return
        it = sel[0]
        for p, k, e in self.edges:
            if it.key in (p, k):
                e.setPen(QPen(QColor("#7aa2ff"), 2.4))
                e.setZValue(1.5)
        m = self.model
        if isinstance(it.key, tuple):
            kind = "Вход модели" if it.key[0] == "in" else "Выход модели"
            self.detail.setHtml(f"<h3>{kind}</h3><p><code>{it.key[1]}</code></p>"
                                + ("<p>Сюда подаётся картинка (нормализованная 0..1, порядок каналов RGB, формат NCHW).</p>"
                                   if it.key[0] == "in" else "<p>Здесь модель выдаёт результат. " + O.describe_outputs(m) + "</p>"))
            return
        n = m.nodes[it.key]
        title, doc = O.OP_DOC.get(n.op, (n.op, "Служебная или редкая операция."))
        wts = []
        for i in n.inputs:
            if i in m.inits:
                wts.append(f"<code>{i}</code> {m.inits[i][1]} {m.inits[i][0]}")
        attrs = "".join(f"<li><code>{k}</code> = {v}</li>" for k, v in n.attrs.items())
        extra = ""
        if n.op == "Conv" and "kernel_shape" in n.attrs:
            k = n.attrs["kernel_shape"]
            st = n.attrs.get("strides", [1, 1])
            extra = (f"<p>Окно {k[0]}×{k[1]}" + (f", шаг {st[0]} - карта признаков уменьшается в {st[0]} раза" if st[0] > 1 else "")
                     + (f", групп: {n.attrs['group']} (поканальная свёртка)" if n.attrs.get("group", 1) > 1 else "") + ".</p>")
        self.detail.setHtml(
            f"<h3>{n.op} - {title}</h3><p>{doc}</p>{extra}"
            f"<p><b>Имя:</b> <code>{n.name}</code> (№{n.idx})</p>"
            + (f"<p><b>Веса:</b><br>{'<br>'.join(wts)}</p>" if wts else "")
            + (f"<p><b>Параметры:</b></p><ul>{attrs}</ul>" if attrs else ""))
