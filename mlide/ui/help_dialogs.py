"""Вспомогательные окна: помощь с видеокартой (CUDA) и примеры кода для запуска обученной модели."""
from __future__ import annotations

import re
import shutil
import subprocess

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from .learn import cards_html


def detect_gpu() -> dict:
    """Читает nvidia-smi. Возвращает {found, name, driver, cuda, raw}."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"found": False}
    try:
        q = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=15)
        full = subprocess.run([exe], capture_output=True, text=True, timeout=15).stdout
        name, driver, mem = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        m = re.search(r"CUDA Version:\s*([\d.]+)", full)
        return {"found": True, "name": name, "driver": driver, "mem": mem, "cuda": m.group(1) if m else "?"}
    except Exception as e:  # noqa
        return {"found": False, "error": str(e)}


class GpuHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Видеокарта и ускорение обучения")
        self.resize(760, 600)
        lay = QVBoxLayout(self)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        site = QPushButton("Открыть pytorch.org: выбор сборки")
        site.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://pytorch.org/get-started/locally/")))
        close = QPushButton("Закрыть")
        close.clicked.connect(self.accept)
        row.addWidget(site)
        row.addWidget(close)
        lay.addLayout(row)
        self._fill()

    def _fill(self):
        g = detect_gpu()
        cards = []
        if g.get("found"):
            cards.append(("ok", f"Найдена видеокарта Nvidia: {g['name']}",
                          f"Память {g['mem']}, драйвер {g['driver']}, поддерживаемая версия CUDA до {g['cuda']}."))
            cards.append(("warn", "Но обучение может идти на процессоре",
                          "Обычная установка <code>pip install ultralytics</code> на Windows ставит PyTorch <b>без поддержки видеокарты</b> "
                          "(в названии версии есть <code>+cpu</code>). Проверка: в окне «Обучение» нажмите «Проверить» - должно быть "
                          "«CUDA доступна: True»."))
            cards.append(("info", "Как включить видеокарту",
                          "1. Откройте страницу выбора сборки (кнопка внизу): Stable → Windows → Pip → Python → <b>CUDA</b> "
                          f"(выберите версию не выше {g['cuda']}).<br>"
                          "2. Сайт покажет команду вида <code>pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX</code>.<br>"
                          "3. Сначала удалите CPU-версию: <code>pip uninstall -y torch torchvision</code>, затем выполните команду с сайта "
                          "в том же Python, который указан в окне «Обучение».<br>"
                          "4. В «Обучении» нажмите «Проверить» - убедитесь, что CUDA доступна. Дальше YOLO сама использует видеокарту "
                          "(batch можно увеличить до 32-64 или поставить -1)."))
        else:
            cards.append(("info", "Видеокарта Nvidia не найдена",
                          "Команда <code>nvidia-smi</code> недоступна: либо видеокарты Nvidia нет, либо не установлен драйвер. "
                          "Обучение пойдёт на процессоре - для небольших датасетов (до нескольких сотен картинок, модель n) это терпимо."))
            cards.append(("info", "Бесплатные варианты с GPU",
                          "<b>Google Colab</b> и <b>Kaggle Notebooks</b> дают бесплатную видеокарту. Экспортируйте датасет из IDE (формат Ultralytics), "
                          "загрузите папку вместе с <code>train.py</code> и запустите там <code>python train.py</code>; готовый <code>best.pt</code> "
                          "скачайте обратно и откройте результаты в IDE.<br>AMD/Intel: возможны OpenVINO (Intel) и DirectML, но "
                          "обучение на них менее надёжно - для обучения проще Nvidia или облако."))
        self.view.setHtml(cards_html(cards))


# --------------------------------------------------------------------------- примеры кода
def snippets(weights_name: str, names: list[str], imgsz: int = 640) -> dict[str, str]:
    base = weights_name.rsplit(".", 1)[0]
    nm = repr(list(names))
    ultra = f'''# pip install ultralytics
from ultralytics import YOLO

model = YOLO("{weights_name}")                 # .pt или .onnx
results = model.predict("photo.jpg", conf=0.25, imgsz={imgsz})
for r in results:
    for b in r.boxes:
        print(model.names[int(b.cls)], round(float(b.conf), 2), [round(v) for v in b.xyxy[0].tolist()])
    r.save("result.jpg")                      # картинка с нарисованными рамками
'''
    cv = f'''# Запуск ONNX-модели YOLO (детекция) без PyTorch.   pip install opencv-python numpy
# python detect.py photo.jpg
import sys

import cv2
import numpy as np

MODEL = "{base}.onnx"
NAMES = {nm}
IMGSZ, CONF, IOU = {imgsz}, 0.25, 0.45

net = cv2.dnn.readNetFromONNX(MODEL)


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return canvas, r, left, top


def detect(img):
    lb, r, left, top = letterbox(img, IMGSZ)
    net.setInput(cv2.dnn.blobFromImage(lb, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True))
    out = net.forward()[0].T                    # (кандидаты, 4 + число классов)
    scores = out[:, 4:]
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf >= CONF
    out, cls, conf = out[keep], cls[keep], conf[keep]
    boxes = np.c_[out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 2], out[:, 3]]   # x, y, w, h
    result = []
    if len(boxes):
        for i in np.array(cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), CONF, IOU)).reshape(-1):
            x, y, w, h = boxes[i]
            x1, y1 = (x - left) / r, (y - top) / r
            result.append((NAMES[cls[i]], float(conf[i]), (x1, y1, x1 + w / r, y1 + h / r)))
    return result


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    for name, conf, (x1, y1, x2, y2) in detect(img):
        print(f"{{name}} {{conf:.2f}} ({{x1:.0f}}, {{y1:.0f}}, {{x2:.0f}}, {{y2:.0f}})")
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{{name}} {{conf:.2f}}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite("result.jpg", img)
'''
    cam = f'''# Камера в реальном времени (на базе detect.py из соседней вкладки).   Выход - клавиша Q
import cv2
from detect import detect

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    for name, conf, (x1, y1, x2, y2) in detect(frame):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{{name}} {{conf:.2f}}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("model", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
'''
    return {"Python · ultralytics": ultra, "Python · OpenCV (ONNX)": cv, "Python · камера": cam}


class CodeSnippetsDialog(QDialog):
    def __init__(self, weights_name: str, names: list[str], imgsz: int = 640, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Примеры кода для запуска модели")
        self.resize(860, 640)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Готовые скрипты: скопируйте, положите рядом с файлом модели и запустите. Названия классов подставлены "
                             "из вашего проекта."))
        self.tabs = QTabWidget()
        self.boxes: dict[str, QPlainTextEdit] = {}
        for title, code in snippets(weights_name, names, imgsz).items():
            w = QWidget()
            v = QVBoxLayout(w)
            ed = QPlainTextEdit(code)
            ed.setReadOnly(True)
            ed.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
            b = QPushButton("Скопировать")
            b.clicked.connect(lambda _=False, e=ed: QGuiApplication.clipboard().setText(e.toPlainText()))
            v.addWidget(ed, 1)
            v.addWidget(b)
            self.tabs.addTab(w, title)
            self.boxes[title] = ed
        lay.addWidget(self.tabs, 1)
        note = QLabel("<span style='color:#8b92a1'>Для C++, C#, Java и мобильных платформ используйте тот же ONNX-файл: "
                      "алгоритм подготовки картинки (letterbox, нормализация 0..1, RGB) и разбора ответа (кандидаты → NMS) "
                      "остаётся тем же, что в скрипте на вкладке OpenCV.</span>")
        note.setWordWrap(True)
        lay.addWidget(note)
