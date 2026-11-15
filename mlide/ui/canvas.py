"""Image canvas: zoom/pan, draw boxes & polygons, select/move/resize, suggestions."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QFont, QImage, QPainter, QPen, QPolygonF)
from PySide6.QtWidgets import QWidget

from ..model import Annotation, ImageEntry

HANDLE = 5          # px radius of handles (view space)


class Canvas(QWidget):
    beforeChange = Signal()          # about to mutate annotations (push undo)
    changed = Signal()               # annotations mutated
    selectionChanged = Signal(object)
    cursorMoved = Signal(float, float)
    zoomChanged = Signal(float)
    smartBox = Signal(float, float, float, float)
    samPoints = Signal(object)         # список (x, y, метка) для SAM

    def __init__(self, project_getter, parent=None):
        super().__init__(parent)
        self.get_project = project_getter
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(300, 300)
        self.image: QImage | None = None
        self.entry: ImageEntry | None = None
        self.tool = "select"                 # select | box | poly | smart
        self.cur_class = 0
        self.scale = 1.0
        self.off = QPointF(0, 0)
        self.selected: str | None = None
        self.show_labels = True
        self.show_fill = True
        self._mouse = QPointF()
        self._pan = None
        self._space = False
        self._drag = None                    # dict describing active drag
        self._poly: list[list[float]] = []   # in-progress polygon (image coords)
        self._pending_undo = False
        self._sam_pts: list = []
        self._sam_poly: list | None = None
        self._fitted = True

    # ------------------------------------------------------------------ public
    def set_entry(self, entry: ImageEntry | None, image: QImage | None):
        self.entry, self.image = entry, image
        self.selected = None
        self._poly = []
        self._drag = None
        self.cancel_sam()
        self.fit()
        self.selectionChanged.emit(None)

    def fit(self):
        if not self.image:
            self.update()
            return
        self._fitted = True
        iw, ih = self.image.width(), self.image.height()
        self.scale = max(0.01, min(self.width() / iw, self.height() / ih) * 0.97)
        self.off = QPointF((self.width() - iw * self.scale) / 2, (self.height() - ih * self.scale) / 2)
        self.zoomChanged.emit(self.scale)
        self.update()

    def zoom_to(self, s, anchor=None):
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        s = min(max(s, 0.02), 64)
        self._fitted = False
        ip = self.to_img(anchor)
        self.scale = s
        self.off = QPointF(anchor.x() - ip.x() * s, anchor.y() - ip.y() * s)
        self.zoomChanged.emit(self.scale)
        self.update()

    def select(self, ann_id):
        if self.selected != ann_id:
            self.selected = ann_id
            self.selectionChanged.emit(ann_id)
        self.update()

    def selected_ann(self) -> Annotation | None:
        if not self.entry or not self.selected:
            return None
        return next((a for a in self.entry.anns if a.id == self.selected), None)

    def delete_selected(self):
        a = self.selected_ann()
        if a:
            self.beforeChange.emit()
            self.entry.anns.remove(a)
            self.select(None)
            self.changed.emit()

    def cancel_poly(self):
        self._poly = []
        self.cancel_sam()
        self.update()

    def cancel_sam(self):
        self._sam_pts = []
        self._sam_poly = None
        self.update()

    def set_sam_preview(self, poly):
        self._sam_poly = poly
        self.update()

    def accept_sam(self, as_box=False):
        if not self._sam_poly or not self.entry:
            return False
        if as_box:
            xs = [p[0] for p in self._sam_poly]
            ys = [p[1] for p in self._sam_poly]
            ann = Annotation(self.cur_class, "box", [[min(xs), min(ys)], [max(xs), max(ys)]])
        else:
            ann = Annotation(self.cur_class, "poly", [list(p) for p in self._sam_poly])
        self.beforeChange.emit()
        self.entry.anns.append(ann)
        self.selected = ann.id
        self.selectionChanged.emit(ann.id)
        self._sam_pts, self._sam_poly = [], None
        self.changed.emit()
        self.update()
        return True

    # ------------------------------------------------------------------ coords
    def to_img(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self.off.x()) / self.scale, (p.y() - self.off.y()) / self.scale)

    def to_view(self, x, y) -> QPointF:
        return QPointF(x * self.scale + self.off.x(), y * self.scale + self.off.y())

    def _clamp(self, p: QPointF) -> QPointF:
        return QPointF(min(max(p.x(), 0), self.image.width()), min(max(p.y(), 0), self.image.height()))

    def _color(self, cls) -> QColor:
        pr = self.get_project()
        if 0 <= cls < len(pr.classes):
            return QColor(pr.classes[cls]["color"])
        return QColor("#aaaaaa")

    def _cname(self, cls) -> str:
        pr = self.get_project()
        return pr.classes[cls]["name"] if 0 <= cls < len(pr.classes) else f"#{cls}"

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 25, 28))
        if not self.image or not self.entry:
            p.setPen(QColor(130, 134, 142))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect(), Qt.AlignCenter, "Импортируйте изображения, чтобы начать\n(Файл > Импорт изображений… или перетащите файлы сюда)")
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 1)
        p.save()
        p.translate(self.off)
        p.scale(self.scale, self.scale)
        p.drawImage(0, 0, self.image)
        p.restore()
        p.setRenderHint(QPainter.Antialiasing, True)

        for a in self.entry.anns:
            self._paint_ann(p, a)

        # in-progress polygon
        if self._poly:
            pts = [self.to_view(*q) for q in self._poly]
            col = self._color(self.cur_class)
            p.setPen(QPen(col, 2))
            p.drawPolyline(QPolygonF(pts))
            p.drawLine(pts[-1], self._mouse)
            p.setBrush(QBrush(col))
            for i, q in enumerate(pts):
                p.drawEllipse(q, 6 if i == 0 else 3.5, 6 if i == 0 else 3.5)

        # превью SAM: маска и точки
        if self._sam_poly:
            col = self._color(self.cur_class)
            p.setPen(QPen(col, 2))
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 90))
            p.drawPolygon(QPolygonF([self.to_view(x, y) for x, y in self._sam_poly]))
        for (x, y, lb) in self._sam_pts:
            q = self.to_view(x, y)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(QColor(60, 220, 110) if lb else QColor(240, 70, 70))
            p.drawEllipse(q, 6, 6)
        # rubber band for box/smart
        if self._drag and self._drag["mode"] == "draw":
            a, b = self._drag["start"], self._drag["cur"]
            r = QRectF(self.to_view(a.x(), a.y()), self.to_view(b.x(), b.y())).normalized()
            col = self._color(self.cur_class)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 40))
            p.drawRect(r)

        # crosshair
        if self.tool in ("box", "smart", "poly", "sam") and self.underMouse() and not self._pan:
            p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.DotLine))
            p.drawLine(0, int(self._mouse.y()), self.width(), int(self._mouse.y()))
            p.drawLine(int(self._mouse.x()), 0, int(self._mouse.x()), self.height())

    def _paint_ann(self, p: QPainter, a: Annotation):
        col = self._color(a.cls)
        sel = a.id == self.selected
        pen = QPen(col, 3 if sel else 2)
        if a.suggested:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        fill_a = 70 if sel else (45 if self.show_fill else 0)
        p.setBrush(QColor(col.red(), col.green(), col.blue(), fill_a))
        vpts = [self.to_view(x, y) for x, y in a.polygon()]
        if a.kind == "box":
            x1, y1, x2, y2 = a.bbox()
            r = QRectF(self.to_view(x1, y1), self.to_view(x2, y2))
            p.drawRect(r)
            anchor = r.topLeft()
        else:
            p.drawPolygon(QPolygonF(vpts))
            anchor = min(vpts, key=lambda q: (q.y(), q.x()))
        if self.show_labels:
            txt = self._cname(a.cls) + (f" {a.conf:.2f}" if a.conf is not None else "") + (" ?" if a.suggested else "")
            p.setFont(QFont("Segoe UI", 9))
            fm = p.fontMetrics()
            tw, th = fm.horizontalAdvance(txt) + 8, fm.height() + 2
            bg = QRectF(anchor.x(), anchor.y() - th, tw, th)
            p.fillRect(bg, col)
            p.setPen(QColor(0, 0, 0) if col.lightness() > 140 else QColor(255, 255, 255))
            p.drawText(bg, Qt.AlignCenter, txt)
        if sel:
            p.setPen(QPen(QColor(255, 255, 255), 1.5))
            p.setBrush(col)
            for q in self._handles(a):
                p.drawRect(QRectF(q.x() - HANDLE, q.y() - HANDLE, HANDLE * 2, HANDLE * 2))

    # ------------------------------------------------------------------ geometry / hit testing
    def _handles(self, a: Annotation):
        if a.kind == "box":
            x1, y1, x2, y2 = a.bbox()
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            order = [(x1, y1), (mx, y1), (x2, y1), (x2, my), (x2, y2), (mx, y2), (x1, y2), (x1, my)]
            return [self.to_view(x, y) for x, y in order]
        return [self.to_view(x, y) for x, y in a.pts]

    def _hit_handle(self, a, pos: QPointF):
        for i, q in enumerate(self._handles(a)):
            if abs(q.x() - pos.x()) <= HANDLE + 3 and abs(q.y() - pos.y()) <= HANDLE + 3:
                return i
        return None

    def _contains(self, a: Annotation, ip: QPointF) -> bool:
        if a.kind == "box":
            x1, y1, x2, y2 = a.bbox()
            return x1 <= ip.x() <= x2 and y1 <= ip.y() <= y2
        return QPolygonF([QPointF(x, y) for x, y in a.pts]).containsPoint(ip, Qt.OddEvenFill)

    def _area(self, a):
        x1, y1, x2, y2 = a.bbox()
        return (x2 - x1) * (y2 - y1)

    def _pick(self, ip: QPointF):
        hits = [a for a in self.entry.anns if self._contains(a, ip)]
        return min(hits, key=self._area) if hits else None

    def _edge_hit(self, a, ip):
        """index i of polygon edge (i -> i+1) near point ip, else None"""
        pts = a.pts
        tol = 6 / self.scale
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            dx, dy = x2 - x1, y2 - y1
            L = dx * dx + dy * dy or 1e-9
            t = max(0, min(1, ((ip.x() - x1) * dx + (ip.y() - y1) * dy) / L))
            if math.hypot(ip.x() - (x1 + t * dx), ip.y() - (y1 + t * dy)) <= tol:
                return i, QPointF(x1 + t * dx, y1 + t * dy)
        return None

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        if e.button() in (Qt.MiddleButton, Qt.RightButton) or (e.button() == Qt.LeftButton and self._space):
            self._pan = (pos, QPointF(self.off))
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() != Qt.LeftButton or not self.image:
            return
        ip = self._clamp(self.to_img(pos))
        if self.tool == "sam":
            lb = 0 if (e.modifiers() & Qt.ControlModifier) else 1
            self._sam_pts.append((ip.x(), ip.y(), lb))
            self.update()
            self.samPoints.emit(list(self._sam_pts))
            return
        if self.tool in ("box", "smart"):
            self._drag = {"mode": "draw", "start": ip, "cur": ip}
        elif self.tool == "poly":
            if len(self._poly) >= 3:
                f = self.to_view(*self._poly[0])
                if math.hypot(f.x() - pos.x(), f.y() - pos.y()) < 10:
                    self._finish_poly()
                    return
            self._poly.append([ip.x(), ip.y()])
            self.update()
        else:
            self._press_select(pos, ip, e.modifiers())

    def _press_select(self, pos, ip, mods):
        a = self.selected_ann()
        if a:
            h = self._hit_handle(a, pos)
            if h is not None:
                if a.kind == "poly" and (mods & Qt.AltModifier) and len(a.pts) > 3:
                    self.beforeChange.emit()
                    a.pts.pop(h)
                    self.changed.emit()
                    return
                self._drag = {"mode": "handle", "idx": h, "undo": True}
                return
        hit = self._pick(ip)
        self.select(hit.id if hit else None)
        if hit:
            self._drag = {"mode": "move", "last": ip, "undo": True}
        self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.LeftButton or not self.image:
            return
        if self.tool == "poly" and len(self._poly) >= 3:
            self._poly.pop()                      # the press already added this point
            self._finish_poly()
        elif self.tool == "select":
            a = self.selected_ann()
            if a and a.kind == "poly":
                hit = self._edge_hit(a, self.to_img(e.position()))
                if hit:
                    self.beforeChange.emit()
                    a.pts.insert(hit[0] + 1, [hit[1].x(), hit[1].y()])
                    self.changed.emit()

    def mouseMoveEvent(self, e):
        pos = e.position()
        self._mouse = pos
        if self._pan:
            start, off0 = self._pan
            self.off = off0 + (pos - start)
            self.update()
            return
        if not self.image:
            return
        ip = self.to_img(pos)
        self.cursorMoved.emit(ip.x(), ip.y())
        d = self._drag
        if d:
            c = self._clamp(ip)
            if d["mode"] == "draw":
                d["cur"] = c
            elif d["mode"] == "move":
                a = self.selected_ann()
                if a:
                    dx, dy = c.x() - d["last"].x(), c.y() - d["last"].y()
                    xs = [q[0] for q in a.pts]
                    ys = [q[1] for q in a.pts]
                    W, H = self.image.width(), self.image.height()
                    dx = max(-min(xs), min(W - max(xs), dx))
                    dy = max(-min(ys), min(H - max(ys), dy))
                    if dx or dy:
                        self._push_undo_once(d)
                        for q in a.pts:
                            q[0] += dx
                            q[1] += dy
                        d["last"] = QPointF(d["last"].x() + dx, d["last"].y() + dy)
                        self.changed.emit()
            elif d["mode"] == "handle":
                a = self.selected_ann()
                if a:
                    self._push_undo_once(d)
                    self._move_handle(a, d["idx"], c)
                    self.changed.emit()
        elif self.tool == "select":
            a = self.selected_ann()
            if a and self._hit_handle(a, pos) is not None:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.SizeAllCursor if self._pick(ip) else Qt.ArrowCursor)
        elif self.tool in ("box", "smart", "poly", "sam"):
            self.setCursor(Qt.CrossCursor)
        self.update()

    def _push_undo_once(self, d):
        if d.get("undo"):
            d["undo"] = False
            self.beforeChange.emit()

    def _move_handle(self, a: Annotation, idx, c: QPointF):
        if a.kind == "poly":
            a.pts[idx] = [c.x(), c.y()]
            return
        x1, y1, x2, y2 = a.bbox()
        if idx in (0, 6, 7):
            x1 = c.x()
        if idx in (2, 3, 4):
            x2 = c.x()
        if idx in (0, 1, 2):
            y1 = c.y()
        if idx in (4, 5, 6):
            y2 = c.y()
        a.pts = [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]]

    def mouseReleaseEvent(self, e):
        if self._pan:
            self._pan = None
            self.setCursor(Qt.ArrowCursor if self.tool == "select" else Qt.CrossCursor)
            return
        d, self._drag = self._drag, None
        if e.button() != Qt.LeftButton or not d:
            return
        if d["mode"] == "draw":
            a, b = d["start"], d["cur"]
            x1, y1, x2, y2 = min(a.x(), b.x()), min(a.y(), b.y()), max(a.x(), b.x()), max(a.y(), b.y())
            if (x2 - x1) * self.scale >= 4 and (y2 - y1) * self.scale >= 4:
                if self.tool == "smart":
                    self.smartBox.emit(x1, y1, x2, y2)
                else:
                    self.beforeChange.emit()
                    ann = Annotation(self.cur_class, "box", [[x1, y1], [x2, y2]])
                    self.entry.anns.append(ann)
                    self.selected = ann.id
                    self.selectionChanged.emit(ann.id)
                    self.changed.emit()
        self.update()

    def _finish_poly(self):
        if len(self._poly) >= 3:
            self.beforeChange.emit()
            ann = Annotation(self.cur_class, "poly", [list(q) for q in self._poly])
            self.entry.anns.append(ann)
            self.selected = ann.id
            self.selectionChanged.emit(ann.id)
            self.changed.emit()
        self._poly = []
        self.update()

    def add_polygon(self, pts, cls=None):
        """Used by smart tool."""
        self.beforeChange.emit()
        ann = Annotation(self.cur_class if cls is None else cls, "poly", pts)
        self.entry.anns.append(ann)
        self.selected = ann.id
        self.selectionChanged.emit(ann.id)
        self.changed.emit()

    def wheelEvent(self, e):
        if not self.image:
            return
        f = 1.18 ** (e.angleDelta().y() / 120)
        self.zoom_to(self.scale * f, e.position())

    def resizeEvent(self, e):
        if self.image and self._fitted:
            self.fit()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Space:
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        elif k in (Qt.Key_Return, Qt.Key_Enter) and self._poly:
            self._finish_poly()
        elif k in (Qt.Key_Return, Qt.Key_Enter) and self.tool == "sam" and self._sam_poly:
            self.accept_sam(as_box=bool(e.modifiers() & Qt.ShiftModifier))
        elif k == Qt.Key_Escape:
            self.cancel_sam()
            self._poly = []
            self._drag = None
            self.select(None)
            self.update()
        elif k == Qt.Key_Backspace and self._poly:
            self._poly.pop()
            self.update()
        elif k == Qt.Key_Backspace and self.tool == "sam" and self._sam_pts:
            self._sam_pts.pop()
            if self._sam_pts:
                self.samPoints.emit(list(self._sam_pts))
            else:
                self._sam_poly = None
            self.update()
        elif k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and self.selected_ann() and not e.modifiers():
            a = self.selected_ann()
            dx = {Qt.Key_Left: -1, Qt.Key_Right: 1}.get(k, 0)
            dy = {Qt.Key_Up: -1, Qt.Key_Down: 1}.get(k, 0)
            self.beforeChange.emit()
            for q in a.pts:
                q[0] += dx
                q[1] += dy
            self.changed.emit()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space = False
            self.setCursor(Qt.ArrowCursor if self.tool == "select" else Qt.CrossCursor)
        super().keyReleaseEvent(e)
