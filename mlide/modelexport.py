"""Экспорт обученных моделей (.pt) в форматы для развёртывания. Работает через внешний Python с ultralytics."""
from __future__ import annotations

import json

# key = значение параметра format в ultralytics
MODEL_FORMATS = [
    dict(key="onnx", title="ONNX", ext=".onnx", opts=("half", "dynamic", "simplify"),
         where="Универсальный формат: Windows, Linux, macOS, веб. Запускается без PyTorch через onnxruntime, OpenCV (cv2.dnn), "
               "C++, C#, Java. <b>Можно загрузить обратно в «Авторазметку» этой IDE.</b>",
         tip="Лучший выбор по умолчанию.",
         next="Файл .onnx можно загрузить в <i>Инструменты → Авторазметка</i> или запускать в своём приложении "
              "(onnxruntime, OpenCV)."),
    dict(key="torchscript", title="TorchScript", ext=".torchscript", opts=(),
         where="Для C++ (LibTorch) и мобильных приложений на PyTorch: модель запускается без исходного Python-кода.",
         tip="Нужен PyTorch на устройстве.",
         next="Файл .torchscript загружается в C++ через torch::jit::load или в Python через torch.jit.load."),
    dict(key="openvino", title="OpenVINO (Intel)", ext="_openvino_model", opts=("half", "int8"),
         where="Быстрый запуск на процессорах и видеокартах Intel; хорошо подходит для обычных ПК и мини-ПК без Nvidia.",
         tip="Создаётся папка. int8 даёт ускорение, но чуть снижает точность.",
         next="Результат - папка с .xml и .bin; запускается через пакет openvino."),
    dict(key="engine", title="TensorRT (Nvidia)", ext=".engine", opts=("half", "int8", "dynamic"),
         where="Максимальная скорость на видеокартах Nvidia (реальное время, камеры, Jetson). "
               "<b>Нужна видеокарта Nvidia и установленный TensorRT.</b> Файл собирается под конкретную карту.",
         tip="На компьютере без Nvidia не получится.",
         next="Файл .engine работает только на видеокарте того же поколения, на которой собран."),
    dict(key="ncnn", title="NCNN (мобильные, Raspberry Pi)", ext="_ncnn_model", opts=("half",),
         where="Лёгкий движок для Android, iOS, Raspberry Pi и встраиваемых устройств без GPU.",
         tip="Создаётся папка с .param и .bin.",
         next="Папка с .param и .bin подключается через библиотеку ncnn."),
    dict(key="tflite", title="TensorFlow Lite (Android)", ext=".tflite", opts=("half", "int8"),
         where="Android-приложения, микроконтроллеры, Google Coral (с int8). "
               "<b>Нужен TensorFlow: тяжёлая зависимость, на новых версиях Python может не установиться.</b>",
         tip="Если установка падает - возьмите Python 3.10-3.11 или используйте ONNX/NCNN.",
         next="Файл .tflite кладётся в assets Android-приложения."),
    dict(key="coreml", title="CoreML (iPhone / Mac)", ext=".mlpackage", opts=("half", "int8"),
         where="Приложения для iPhone, iPad и Mac. <b>Экспорт надёжнее всего работает на macOS.</b>",
         tip="На Windows может быть недоступен.",
         next="Пакет .mlpackage добавляется в проект Xcode."),
]
MODEL_FORMATS.insert(1, dict(
    key="esp32", fmt="onnx", title="ESP32-S3 / ESP32-P4 (через ESP-DL)", ext=".onnx", opts=("simplify",), imgsz=320,
    where="Микроконтроллеры Espressif. Здесь готовится <b>ONNX малого размера (320 px)</b> - первый шаг цепочки "
          "ONNX → квантование ESP-PPQ → <code>.espdl</code> → ESP-DL на плате. Поддерживается YOLO11n; "
          "<b>возьмите самую маленькую модель (n) и мало классов</b>.",
    tip="Нужны ESP-IDF ≥ 5.3 и ESP-PPQ; подробный план - Справочник → «Развёртывание на ESP32-S3».",
    next="Дальше (на компьютере с ESP-PPQ): 1) для лучшего качества пере-экспортируйте best.pt скриптом export_onnx.py из "
         "репозитория esp-dl (он переносит декодирование рамок в пост-обработку и снижает ошибку квантования); "
         "2) квантуйте в 8 бит через ESP-PPQ на калибровочных картинках (кнопка «Калибровочные картинки…»); "
         "3) соберите пример yolo11_detect из esp-dl с полученным .espdl. Подробности: Справочник → ESP32-S3."))
BY_KEY = {f["key"]: f for f in MODEL_FORMATS}

OPT_HELP = {
    "half": "FP16: вдвое меньше файл и быстрее на видеокартах/NPU, точность почти не меняется.",
    "int8": "INT8-квантование: файл в 3-4 раза меньше и быстрее на слабом железе, но точность может чуть упасть.",
    "dynamic": "Динамический размер входа: модель принимает картинки разного размера/пачки (чуть медленнее).",
    "simplify": "Упростить граф модели (onnxslim): обычно ускоряет и уменьшает файл.",
}

# код выполняется во внешнем Python; параметры передаются JSON-ом в argv (без подстановки строк в код)
EXPORT_CODE = (
    "import json, sys\n"
    "from ultralytics import YOLO\n"
    "a = json.loads(sys.argv[1])\n"
    "w = a.pop('weights')\n"
    "out = YOLO(w).export(**a)\n"
    "print('RESULT:', out)\n"
)


def build_command(python: str, weights: str, fmt: str, imgsz: int = 640, half=False, int8=False,
                  dynamic=False, simplify=True) -> list[str]:
    spec = BY_KEY[fmt]
    args = {"weights": weights, "format": spec.get("fmt", fmt), "imgsz": int(imgsz)}
    if "half" in spec["opts"] and half:
        args["half"] = True
    if "int8" in spec["opts"] and int8:
        args["int8"] = True
    if "dynamic" in spec["opts"] and dynamic:
        args["dynamic"] = True
    if "simplify" in spec["opts"]:
        args["simplify"] = bool(simplify)
    return [python, "-u", "-c", EXPORT_CODE, json.dumps(args)]


def parse_result(log: str) -> str | None:
    for line in reversed(log.splitlines()):
        if line.startswith("RESULT:"):
            return line.split(":", 1)[1].strip()
    return None


def explain_failure(fmt: str, log: str) -> str:
    low = log.lower()
    if "no module named 'ultralytics'" in low or 'no module named "ultralytics"' in low:
        return "Не установлен пакет ultralytics: в окне «Обучение» нажмите «Установить ultralytics»."
    if fmt == "engine" and ("cuda" in low or "tensorrt" in low):
        return "TensorRT требует видеокарту Nvidia и установленный TensorRT; на этом компьютере экспорт невозможен."
    if fmt == "tflite" and "tensorflow" in low:
        return "Не удалось установить TensorFlow (на новых версиях Python его часто нет). Используйте ONNX или NCNN."
    if fmt == "coreml":
        return "CoreML надёжно экспортируется только на macOS. На Windows используйте ONNX."
    if "out of memory" in low:
        return "Не хватило памяти: уменьшите размер изображения."
    return "Экспорт не удался - подробности в журнале ниже."


def export_calibration(image_paths, out_dir, size: int = 320, n: int = 100, seed: int = 0) -> int:
    """Калибровочные картинки для квантования (ESP-PPQ и др.): n случайных, длинная сторона = size."""
    import random
    from pathlib import Path

    import cv2

    from . import imgio

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = list(image_paths)
    random.Random(seed).shuffle(paths)
    done = 0
    for p in paths[:n]:
        try:
            img = imgio.imread(p)
        except (OSError, ValueError):
            continue                                        # битую картинку для калибровки пропускаем
        h, w = img.shape[:2]
        k = size / max(h, w)
        if k < 1:
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        imgio.imwrite(out / f"calib_{done:04d}.jpg", img)
        done += 1
    return done
