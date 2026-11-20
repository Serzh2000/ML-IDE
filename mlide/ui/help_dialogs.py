"""Вспомогательные окна: помощь с видеокартой (CUDA) и примеры кода для запуска обученной модели."""
from __future__ import annotations

import re
import shutil
import subprocess

from PySide6.QtCore import QProcess, QProcessEnvironment, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import paths, textio

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
        pr = QHBoxLayout()
        self.py = QLineEdit(shutil.which("python") or shutil.which("py") or "")
        self.py.setToolTip("<p>Python, в котором установлен (или будет установлен) PyTorch. Тот же путь указан в окне «Обучение».</p>")
        self.install_btn = QPushButton("Установить PyTorch с CUDA…")
        self.install_btn.setProperty("primary", True)
        self.install_btn.setToolTip("<p>Подберёт сборку под ваш драйвер, заменит версию без GPU и проверит результат. Скачивается 2-3 ГБ.</p>")
        self.install_btn.clicked.connect(self._install)
        pr.addWidget(QLabel("Python:"))
        pr.addWidget(self.py, 1)
        pr.addWidget(self.install_btn)
        lay.addLayout(pr)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.log.setVisible(False)
        lay.addWidget(self.log)
        self.proc = None
        self._steps = []
        self._raw = ""
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

    def _install(self):
        g = detect_gpu()
        if not g.get("found"):
            QMessageBox.information(self, "PyTorch с CUDA", "Видеокарта Nvidia не найдена (nvidia-smi недоступен). "
                                    "Ставить сборку с CUDA нет смысла.")
            return
        py = self.py.text().strip()
        if not py:
            QMessageBox.warning(self, "PyTorch с CUDA", "Укажите путь к Python.")
            return
        self.install_btn.setEnabled(False)
        self.log.setVisible(True)
        self.log.setPlainText("Подбираю подходящую сборку PyTorch…")
        __import__('PySide6.QtWidgets', fromlist=['x']).QApplication.processEvents()
        plan = torch_cuda_plan(g.get("cuda", ""))
        if not plan["ok"]:
            self.log.appendPlainText(plan["reason"])
            self.install_btn.setEnabled(True)
            return
        cmd = f"{py} {' '.join(plan['uninstall'])}\n{py} {' '.join(plan['install'])}"
        if QMessageBox.question(self, "PyTorch с CUDA", f"Будет установлена сборка <b>{plan['tag']}</b> (подходит к вашему драйверу, CUDA до {g['cuda']}).\n\n"
                                f"Команды:\n{cmd}\n\nСкачивается 2-3 ГБ, версия без GPU будет заменена. Продолжить?") != QMessageBox.Yes:
            self.install_btn.setEnabled(True)
            self.log.setVisible(False)
            return
        self._steps = [(py, plan["uninstall"]), (py, plan["install"]),
                       (py, ["-c", "import torch;print('torch', torch.__version__, '| CUDA доступна:', torch.cuda.is_available(), "
                                   "'|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"])]
        self._next_step()

    def _next_step(self):
        if not self._steps:
            self.install_btn.setEnabled(True)
            ok = "CUDA доступна: True" in self._raw[-600:]
            self.log.appendPlainText("\nГотово: видеокарта доступна для обучения." if ok else
                                     "\nВнимание: установка завершена, но CUDA недоступна. Перезапустите компьютер или обновите драйвер Nvidia.")
            return
        py, args = self._steps.pop(0)
        self.log.appendPlainText("\n$ " + py + " " + " ".join(args))
        self.proc = QProcess(self)
        env = paths.child_qenv()
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_out)
        self.proc.finished.connect(lambda code, _s: self._next_step() if (code == 0 or "uninstall" in " ".join(args)) else self._failed(code))
        with paths.child_launch():           # без каталога DLL приложения (иначе Python падает с 0xC0000005)
            self.proc.start(py, args)
            self.proc.waitForStarted(8000)

    def _on_out(self):
        text = textio.decode_stream(self.proc, self.proc.readAllStandardOutput())
        self._raw += text
        self.log.appendPlainText(text.rstrip())

    def _failed(self, code):
        self._steps = []
        self.install_btn.setEnabled(True)
        self.log.appendPlainText(f"\nОшибка: команда завершилась с кодом {code}. Если пакет не удалился или не скачался - повторите позже "
                                 "или выполните команды вручную.")

    def reject(self):
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.proc.waitForFinished(2000)
        super().reject()

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

with open(MODEL, "rb") as f:                  # через байты: работает и с кириллицей в пути
    net = cv2.dnn.readNetFromONNX(np.frombuffer(f.read(), np.uint8))


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


# --------------------------------------------------------------------------- PyTorch с CUDA
CUDA_TAGS = [130, 129, 128, 126, 124, 121, 118]          # индексы https://download.pytorch.org/whl/cuNNN


def parse_cuda(v: str) -> int:
    """'12.8' -> 128, '13.0' -> 130; 0 при ошибке."""
    m = re.match(r"(\d+)\.(\d+)", v or "")
    return int(m.group(1)) * 10 + int(m.group(2)) if m else 0


def probe_index(tag: int, timeout: float = 15.0) -> bool:
    import urllib.request
    try:
        req = urllib.request.Request(f"https://download.pytorch.org/whl/cu{tag}/torch/", method="HEAD",
                                     headers={"User-Agent": "mlide-cuda-helper"})
        return urllib.request.urlopen(req, timeout=timeout).status == 200
    except Exception:
        return False


def torch_cuda_plan(driver_cuda: str, probe=probe_index) -> dict:
    """Какую сборку PyTorch ставить: самая новая из существующих, не выше версии CUDA, которую поддерживает драйвер."""
    limit = parse_cuda(driver_cuda)
    if not limit:
        return {"ok": False, "reason": "Не удалось определить версию CUDA драйвера (nvidia-smi)."}
    for tag in CUDA_TAGS:
        if tag <= limit and probe(tag):
            url = f"https://download.pytorch.org/whl/cu{tag}"
            return {"ok": True, "tag": f"cu{tag}", "index": url,
                    "uninstall": ["-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
                    "install": ["-m", "pip", "install", "torch", "torchvision", "--index-url", url]}
    return {"ok": False, "reason": f"Для CUDA {driver_cuda} не нашлось подходящей сборки PyTorch. Обновите драйвер Nvidia."}


# --------------------------------------------------------------------------- проверка обновлений
class _UpdateWorker(QThread):
    done = Signal(dict)

    def run(self):
        from .. import __version__, updates
        self.done.emit(updates.check(__version__))


class UpdateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from .. import __version__
        self.setWindowTitle("Проверка обновлений")
        self.resize(560, 360)
        self.url = ""
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Установленная версия: <b>{__version__}</b>. Проверка выполняется только по этой кнопке: "
                             "программа делает один обычный запрос к GitHub и ничего о вас не отправляет."))
        lay.itemAt(0).widget().setWordWrap(True)
        self.view = QTextBrowser()
        self.view.setHtml("<p>Проверяю…</p>")
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.open_btn = QPushButton("Открыть страницу релиза")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.url)))
        close = QPushButton("Закрыть")
        close.clicked.connect(self.accept)
        row.addWidget(self.open_btn)
        row.addWidget(close)
        lay.addLayout(row)
        self.worker = _UpdateWorker()
        self.worker.done.connect(self._done)
        self.worker.start()

    def _done(self, r):
        if r["status"] == "error":
            self.view.setHtml(cards_html([("warn", "Не удалось проверить", f"Нет связи с GitHub или он не ответил.<br><code>{r['error']}</code>")]))
            return
        lt = r["latest"]
        self.url = lt["url"]
        self.open_btn.setEnabled(True)
        notes = re.sub(r"<[^>]+>", "", lt["body"] or "").strip().replace("\n", "<br>")[:1500]
        if r["status"] == "new":
            assets = "<br>".join(f"{a['name']} ({a['size'] / 1048576:.0f} МБ)" for a in lt["assets"])
            self.view.setHtml(cards_html([("ok", f"Доступна новая версия {lt['tag']}", (assets + "<br><br>" if assets else "") + notes),
                                          ("info", "Как обновиться", "Скачайте установщик или архив на странице релиза и установите поверх: проекты и настройки сохранятся.")]))
        else:
            self.view.setHtml(cards_html([("ok", "У вас последняя версия", f"Последний релиз на GitHub: {lt['tag']}.")]))

    def reject(self):
        if self.worker.isRunning():
            self.worker.wait(3000)
        super().reject()
