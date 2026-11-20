"""Проверка модели на видео / веб-камере и замер скорости (FPS)."""
from __future__ import annotations

from . import winflags

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox, QTextBrowser, QVBoxLayout)

from .. import autolabel, imgio
from ..model import PALETTE
from .dialogs import add_tips, hex_to_bgr
from . import icons, style
from .learn import cards_html


def class_names_for(model_path: str, project) -> list[str]:
    p = Path(model_path).resolve()
    for up in [p.parent, p.parent.parent, p.parent.parent.parent, p.parent.parent.parent.parent]:
        y = up / "data.yaml"
        if y.exists():
            try:
                n = yaml.safe_load(y.read_text(encoding="utf-8")).get("names")
                return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
            except Exception:
                pass
    return project.class_names()


def draw_dets(img, dets, names):
    out = img
    pal = [hex_to_bgr(c) for c in PALETTE]
    th = max(2, round(max(out.shape[:2]) / 500))
    for d in dets:
        col = pal[d.cls % len(pal)]
        nm = names[d.cls] if d.cls < len(names) else (d.name or str(d.cls))
        x1, y1, x2, y2 = map(int, d.box)
        if d.poly and len(d.poly) >= 3:                       # сегментация / повёрнутая рамка: контур с полупрозрачной заливкой
            poly = np.array(d.poly, np.int32).reshape(-1, 1, 2)
            if not d.obb:
                ov = out.copy()
                cv2.fillPoly(ov, [poly], col)
                cv2.addWeighted(ov, 0.35, out, 0.65, 0, out)
            cv2.polylines(out, [poly], True, col, th)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        for kp in d.kps or []:                                # ключевые точки (pose)
            if len(kp) > 2 and kp[2] > 0:
                cv2.circle(out, (int(kp[0]), int(kp[1])), max(2, th + 1), col, -1)
        cv2.putText(out, f"{nm} {d.conf:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, th / 3), col,
                    max(1, th - 1), cv2.LINE_AA)
    return out


def open_capture(src):
    cap = cv2.VideoCapture(int(src)) if str(src).isdigit() else None
    if cap is None:
        p = str(src)
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():                       # Windows: путь с кириллицей - пробуем короткое имя
            try:
                import ctypes
                buf = ctypes.create_unicode_buffer(1024)
                if ctypes.windll.kernel32.GetShortPathNameW(p, buf, 1024):
                    cap = cv2.VideoCapture(buf.value)
            except Exception:
                pass
    return cap


class _Scan(QThread):
    """Ищет подключённые камеры (номера 0-4)."""
    found = Signal(list)

    def run(self):
        out = []
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    ok, fr = cap.read()
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (fr.shape[1] if ok else 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (fr.shape[0] if ok else 0)
                    out.append((i, f"{w}x{h}" if w and h else ""))
                if cap is not None:
                    cap.release()
            except Exception:
                pass
        self.found.emit(out)


class _Runner(QThread):
    frame = Signal(QImage, float, float)      # картинка, мс на кадр (модель), FPS
    ended = Signal(str)
    dets = Signal(list)                       # для фото: [(имя класса, уверенность), ...]

    def __init__(self, model_path, src, conf, iou, imgsz, names):
        super().__init__()
        self.model_path, self.src, self.conf, self.iou, self.imgsz, self.names = model_path, src, conf, iou, imgsz, names
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa
            self.ended.emit(f"Ошибка во время проверки: {type(e).__name__}: {e}")

    def _run(self):
        try:
            model = autolabel.load_model(self.model_path, self.imgsz)
        except Exception as e:  # noqa
            self.ended.emit(f"Не удалось загрузить модель: {e}")
            return
        if getattr(model, "names", None):                        # названия классов, записанные в самой модели, надёжнее файла рядом
            self.names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if imgio.is_image(self.src):                      # одиночное фото: одна проверка и показ результата
            img = imgio.imread(self.src)
            if img is None:
                self.ended.emit("Не удалось открыть изображение.")
                return
            t0 = time.perf_counter()
            dets = model.predict(img, self.conf, self.iou)
            ms = (time.perf_counter() - t0) * 1000
            vis = draw_dets(img.copy(), dets, self.names)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.dets.emit([(self.names[d.cls] if d.cls < len(self.names) else (d.name or str(d.cls)), d.conf) for d in dets])
            self.frame.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy(), ms, 0.0)
            self.ended.emit("")
            return
        cap = open_capture(self.src)
        if not cap.isOpened():
            self.ended.emit("Не удалось открыть источник видео (проверьте файл или номер камеры).")
            return
        t_prev, ema = time.perf_counter(), None
        try:
            self._loop(cap, model, t_prev, ema)
        finally:
            cap.release()                                    # камера освобождается и при ошибке
        self.ended.emit("")

    def _loop(self, cap, model, t_prev, ema):
        while not self._stop and not self.isInterruptionRequested():
            ok, img = cap.read()
            if not ok:
                if str(self.src).isdigit():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # видеофайл: по кругу
                continue
            t0 = time.perf_counter()
            dets = model.predict(img, self.conf, self.iou)
            ms = (time.perf_counter() - t0) * 1000
            vis = draw_dets(img.copy(), dets, self.names)
            now = time.perf_counter()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            ema = fps if ema is None else 0.9 * ema + 0.1 * fps
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.frame.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy(), ms, ema)


class _Bench(QThread):
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, model_path, imgsz, frame):
        super().__init__()
        self.model_path, self.imgsz, self.frame = model_path, imgsz, frame

    def run(self):
        try:
            model = autolabel.load_model(self.model_path, self.imgsz)
            img = self.frame if self.frame is not None else np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), np.uint8)
            for _ in range(5):
                model.predict(img, 0.25, 0.45)
            ts = []
            for _ in range(60):
                t = time.perf_counter()
                model.predict(img, 0.25, 0.45)
                ts.append((time.perf_counter() - t) * 1000)
            ts.sort()
            self.done.emit({"mean": statistics.mean(ts), "median": statistics.median(ts), "p95": ts[int(len(ts) * 0.95) - 1],
                            "min": ts[0], "n": len(ts)})
        except Exception as e:  # noqa
            self.failed.emit(f"{type(e).__name__}: {e}")


class LiveTestDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Проверка модели на видео и камере")
        self.resize(1000, 780)
        self.project, self.runner, self.bench, self._last = project, None, None, None
        lay = winflags.scroll_layout(self)
        row = QHBoxLayout()
        self.model = QLineEdit(project.export.get("auto_model", ""))
        self.model.setPlaceholderText("Модель .onnx / .pt")
        self._scan = _Scan()
        self._scan.found.connect(self._cams_found)
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse_model)
        row.addWidget(QLabel("Модель:"))
        row.addWidget(self.model, 1)
        row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.src = QComboBox()
        self.src.addItem("Поиск камер…", None)
        self.src.setToolTip("<p>Подключённая камера, либо выбранные вами видеофайл или фото.</p>")
        bs = QPushButton("Видео…")
        bs.setIcon(icons.icon("video"))
        bs.clicked.connect(self._browse_src)
        bp = QPushButton("Фото…")
        bp.setIcon(icons.icon("images"))
        bp.setToolTip("<p>Проверить модель на одной картинке: результат показывается сразу.</p>")
        bp.clicked.connect(self._browse_photo)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.35)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 2048)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        row.addWidget(QLabel("Источник:"))
        row.addWidget(self.src, 1)
        row.addWidget(bs)
        row.addWidget(bp)
        row.addWidget(QLabel("Порог:"))
        row.addWidget(self.conf)
        row.addWidget(QLabel("Размер:"))
        row.addWidget(self.imgsz)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.start = QPushButton("Запустить")
        self.start.setIcon(icons.primary_icon("start"))
        self.start.setProperty("primary", True)
        self.start.clicked.connect(self._start)
        self.stop = QPushButton("Остановить")
        self.stop.setIcon(icons.icon("stop"))
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self._stop)
        self.benchbtn = QPushButton("Замерить скорость")
        self.benchbtn.setIcon(icons.icon("timer"))
        self.benchbtn.clicked.connect(self._bench)
        self.stats = QLabel("")
        row.addWidget(self.start)
        row.addWidget(self.stop)
        row.addWidget(self.benchbtn)
        row.addWidget(self.stats, 1)
        lay.addLayout(row)
        self.view = QLabel("Выберите модель и источник, затем нажмите «Запустить»")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 400)
        self.view.setStyleSheet(f"background:{style.CANVAS};border-radius:8px;")
        lay.addWidget(self.view, 1)
        self.result = QTextBrowser()
        self.result.setMaximumHeight(150)
        lay.addWidget(self.result)
        add_tips(self, {
            self.model: "Модель для проверки: .onnx (работает без torch) или .pt. Для .pt берётся готовый .onnx рядом, а если его нет - файл конвертируется (нужен Python с ultralytics, его ставит окно «Обучение»).",
            self.conf: "Показывать только находки с уверенностью не ниже этого значения.",
            self.imgsz: "Размер входа модели (как при обучении/экспорте). Меньше - быстрее, но мелкие объекты хуже видны.",
        })

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_scanned", False):
            self._scanned = True
            self._scan.start()

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Модель", self.model.text(), "YOLO (*.onnx *.pt)")
        if p:
            self.model.setText(p)

    def _browse_src(self):
        p, _ = QFileDialog.getOpenFileName(self, "Видео", "", "Видео (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;Все файлы (*)")
        if p:
            self._add_source(f"Видео: {Path(p).name}", p)

    def _browse_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Фото", "", "Изображения (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;Все файлы (*)")
        if p:
            self._add_source(f"Фото: {Path(p).name}", p)
            if self.model.text().strip():
                self._start()                               # фото проверяем сразу

    def _add_source(self, label, value):
        i = self.src.findData(value)
        if i < 0:
            self.src.addItem(label, value)
            i = self.src.count() - 1
        self.src.setCurrentIndex(i)

    def _src_value(self):
        v = self.src.currentData()
        return "" if v is None else str(v)

    def _cams_found(self, cams):
        keep = [(self.src.itemText(i), self.src.itemData(i)) for i in range(self.src.count())
                if self.src.itemData(i) is not None and not str(self.src.itemData(i)).isdigit()]
        self.src.clear()
        for i, res in cams:
            self.src.addItem(f"Камера {i}" + (f" ({res})" if res else ""), str(i))
        for t, d in keep:
            self.src.addItem(t, d)
        if self.src.count() == 0:
            self.src.addItem("Камеры не найдены - выберите видео или фото", None)
        self.src.setCurrentIndex(max(0, self.src.count() - 1) if keep else 0)

    def _check(self):
        m = self.model.text().strip()
        if not Path(m).exists():
            QMessageBox.warning(self, "Проверка модели", "Укажите файл модели (.onnx или .pt).")
            return None
        return m

    def _start(self):
        m = self._check()
        if not m:
            return
        self.project.export["auto_model"] = m
        src = self._src_value()
        if not src:
            QMessageBox.information(self, "Проверка модели", "Выберите камеру, видео или фото.")
            return
        if self.runner is not None and self.runner.isRunning():
            self.runner.stop()
            self.runner.wait(3000)
        self.result.clear()
        self.runner = _Runner(m, src, self.conf.value(), 0.45, self.imgsz.value(), class_names_for(m, self.project))
        self.runner.frame.connect(self._frame)
        self.runner.dets.connect(self._photo_dets)
        self.runner.ended.connect(self._ended)
        self.runner.start()
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        self.stats.setText("запуск…")

    def _frame(self, img, ms, fps):
        self._last = img
        self.view.setPixmap(QPixmap.fromImage(img).scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.stats.setText(f"модель: {ms:.0f} мс" + (f"/кадр · показ: {fps:.1f} кадр/с" if fps else ""))

    def _photo_dets(self, items):
        if not items:
            body = "Модель ничего не нашла на этом фото. Попробуйте снизить «Порог» или проверьте другую картинку."
        else:
            cnt = {}
            for n, c in items:
                cnt.setdefault(n, []).append(c)
            body = "; ".join(f"{n}: {len(v)} (уверенность до {max(v):.2f})" for n, v in sorted(cnt.items()))
        self.result.setHtml(cards_html([("ok" if items else "warn", f"Найдено объектов: {len(items)}", body)]))

    def _ended(self, msg):
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        if msg:
            self.stats.setText(f"<span style='color:#ff6b6b'>{msg}</span>")

    def _stop(self):
        if self.runner:
            self.runner.stop()

    def _bench(self):
        m = self._check()
        if not m:
            return
        frame = None
        sv = self._src_value()
        if sv and imgio.is_image(sv):
            frame = imgio.imread(sv)
        elif sv:
            cap = open_capture(sv)
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                cap.release()
                frame = frame if ok else None
        self.benchbtn.setEnabled(False)
        self.stats.setText("замер… (около 10 секунд)")
        self.bench = _Bench(m, self.imgsz.value(), frame)
        self.bench.done.connect(self._bench_done)
        self.bench.failed.connect(lambda e: (self.stats.setText(f"<span style='color:#ff6b6b'>{e}</span>"), self.benchbtn.setEnabled(True)))
        self.bench.start()

    def _bench_done(self, r):
        self.benchbtn.setEnabled(True)
        fps = 1000 / r["mean"]
        self.stats.setText(f"≈ {fps:.1f} кадр/с")
        verdict = ("для камеры 30 кадр/с запаса хватает" if r["mean"] <= 33 else
                   "для камеры 30 кадр/с маловато: уменьшите размер входа (imgsz), возьмите модель меньше (n) или экспортируйте в "
                   "OpenVINO/TensorRT" if fps < 15 else "хватит для задач без жёсткого реального времени")
        self.result.setHtml(cards_html([
            ("info", f"Скорость: {r['mean']:.1f} мс на кадр (≈{fps:.1f} кадр/с)",
             f"Медиана {r['median']:.1f} мс, 95% кадров быстрее {r['p95']:.1f} мс, лучший {r['min']:.1f} мс ({r['n']} запусков, "
             f"вход {self.imgsz.value()} px, процессор). Вывод: {verdict}. Замер включает подготовку картинки и расшифровку ответа."),
            ("info", "Как читать", "На видеокарте/в TensorRT модель обычно в разы быстрее; на ESP32-S3 и подобных устройствах - в десятки раз "
                                   "медленнее, чем на ПК.")]))

    def closeEvent(self, e):
        self._scan.wait(3000)
        self._stop()
        if self.runner:
            self.runner.wait(2000)
        super().closeEvent(e)

    def reject(self):
        self._stop()
        if self.runner:
            self.runner.wait(2000)
        super().reject()
