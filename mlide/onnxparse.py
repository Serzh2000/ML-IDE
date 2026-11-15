"""Минимальный разбор ONNX (protobuf) без пакета onnx: граф, формы, параметры, атрибуты."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

DTYPES = {1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64", 8: "string",
          9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16"}


# ----------------------------------------------------------------------------- protobuf wire format
def _varint(b: bytes, i: int):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def _fields(b: bytes):
    """Итератор (номер поля, тип, значение): varint -> int, LEN -> bytes, fixed32/64 -> bytes."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        else:
            raise ValueError("неподдерживаемый тип поля protobuf")
        yield f, w, v


def _packed_varints(b: bytes):
    i, out = 0, []
    while i < len(b):
        v, i = _varint(b, i)
        out.append(v)
    return out


def _s(v):
    return v.decode("utf-8", "replace")


def _signed(v):
    return v - (1 << 64) if v >= (1 << 63) else v


# ----------------------------------------------------------------------------- модель данных
@dataclass
class Node:
    idx: int
    name: str
    op: str
    inputs: list
    outputs: list
    domain: str = ""
    attrs: dict = field(default_factory=dict)


@dataclass
class OnnxModel:
    path: str = ""
    size: int = 0
    ir_version: int = 0
    opsets: dict = field(default_factory=dict)
    producer: str = ""
    graph_name: str = ""
    nodes: list = field(default_factory=list)
    inputs: list = field(default_factory=list)        # (name, dtype, shape)
    outputs: list = field(default_factory=list)
    inits: dict = field(default_factory=dict)         # name -> (dims, dtype)

    @property
    def params(self) -> int:
        total = 0
        for dims, _ in self.inits.values():
            p = 1
            for d in dims:
                p *= max(d, 1)
            total += p
        return total


def _parse_attr(b: bytes):
    name, val, typ = "", None, 0
    ints, floats = [], []
    for f, w, v in _fields(b):
        if f == 1:
            name = _s(v)
        elif f == 20:
            typ = v
        elif f == 2 and w == 5:
            val = round(struct.unpack("<f", v)[0], 6)
        elif f == 3:
            val = _signed(v)
        elif f == 4:
            val = _s(v)
        elif f == 7:
            floats += list(struct.unpack(f"<{len(v) // 4}f", v)) if w == 2 else [struct.unpack("<f", v)[0]]
        elif f == 8:
            ints += [_signed(x) for x in _packed_varints(v)] if w == 2 else [_signed(v)]
    if ints:
        val = ints
    elif floats:
        val = [round(x, 6) for x in floats]
    elif typ == 4 and val is None:
        val = "<тензор>"
    return name, val


def _parse_node(b: bytes, idx: int) -> Node:
    n = Node(idx, "", "", [], [])
    for f, w, v in _fields(b):
        if f == 1:
            n.inputs.append(_s(v))
        elif f == 2:
            n.outputs.append(_s(v))
        elif f == 3:
            n.name = _s(v)
        elif f == 4:
            n.op = _s(v)
        elif f == 7:
            n.domain = _s(v)
        elif f == 5:
            k, val = _parse_attr(v)
            if val is not None:
                n.attrs[k] = val
    if not n.name:
        n.name = f"{n.op}_{idx}"
    return n


def _parse_tensor(b: bytes):
    name, dims, dt = "", [], 0
    for f, w, v in _fields(b):
        if f == 1:
            dims += [_signed(x) for x in _packed_varints(v)] if w == 2 else [_signed(v)]
        elif f == 2:
            dt = v
        elif f == 8:
            name = _s(v)
    return name, dims, dt


def _parse_value_info(b: bytes):
    name, dt, shape = "", 0, []
    for f, w, v in _fields(b):
        if f == 1:
            name = _s(v)
        elif f == 2:                                   # TypeProto
            for f2, _w2, v2 in _fields(v):
                if f2 == 1:                            # tensor_type
                    for f3, _w3, v3 in _fields(v2):
                        if f3 == 1:
                            dt = v3
                        elif f3 == 2:                  # shape
                            for f4, _w4, v4 in _fields(v3):
                                if f4 == 1:            # dim
                                    d = "?"
                                    for f5, _w5, v5 in _fields(v4):
                                        if f5 == 1:
                                            d = _signed(v5)
                                        elif f5 == 2:
                                            d = _s(v5)
                                    shape.append(d)
    return name, DTYPES.get(dt, str(dt)), shape


def load_onnx(path) -> OnnxModel:
    p = Path(path)
    data = p.read_bytes()
    m = OnnxModel(path=str(p), size=len(data))
    graph = b""
    for f, w, v in _fields(data):
        if f == 1:
            m.ir_version = v
        elif f == 2:
            m.producer = _s(v)
        elif f == 8:
            dom, ver = "", 0
            for f2, _w2, v2 in _fields(v):
                if f2 == 1:
                    dom = _s(v2)
                elif f2 == 2:
                    ver = v2
            m.opsets[dom or "ai.onnx"] = ver
        elif f == 7:
            graph = v
    if not graph:
        raise ValueError("Это не похоже на ONNX-модель (нет графа).")
    for f, w, v in _fields(graph):
        if f == 1:
            m.nodes.append(_parse_node(v, len(m.nodes)))
        elif f == 2:
            m.graph_name = _s(v)
        elif f == 5:
            name, dims, dt = _parse_tensor(v)
            m.inits[name] = (dims, DTYPES.get(dt, str(dt)))
        elif f == 11:
            m.inputs.append(_parse_value_info(v))
        elif f == 12:
            m.outputs.append(_parse_value_info(v))
    m.inputs = [i for i in m.inputs if i[0] not in m.inits]      # старые модели дублируют веса во входах
    return m


# ----------------------------------------------------------------------------- пояснения
OP_DOC = {
    "Conv": ("Свёртка", "Главный «детектор признаков»: скользящим окном (ядро kernel_shape) ищет края, текстуры, формы. "
                         "Веса ядер - это то, чему модель научилась. Чем больше Conv и каналов - тем «умнее» и тяжелее модель."),
    "Sigmoid": ("Сигмоида", "Сжимает число в диапазон 0..1. На выходе - «вероятность»; внутри сети вместе с Mul образует функцию SiLU."),
    "Mul": ("Умножение", "Поэлементное умножение. В паре Conv → Sigmoid → Mul получается активация SiLU (x·sigmoid(x))."),
    "Add": ("Сложение", "Поэлементная сумма. Часто - «короткая связь» (residual): вход блока добавляется к его результату, чтобы глубокая сеть обучалась стабильно."),
    "Concat": ("Склейка", "Объединяет несколько потоков признаков по каналам. Так сеть смешивает информацию разных уровней детализации."),
    "Split": ("Разделение", "Делит тензор на части по каналам - обратная операция к Concat (используется в блоках C2f/C3k2)."),
    "Slice": ("Срез", "Вырезает часть тензора (например, отдельные каналы или координаты)."),
    "MaxPool": ("Макс-пулинг", "Уменьшает карту признаков в несколько раз, оставляя максимум в окне. Даёт устойчивость к сдвигам; в блоке SPPF увеличивает «поле зрения»."),
    "Resize": ("Масштабирование", "Увеличивает карту признаков (upsample), чтобы объединить глубокие признаки с более детальными."),
    "Reshape": ("Изменение формы", "Меняет расположение данных без изменения значений - готовит выход к формату «кандидаты × параметры»."),
    "Transpose": ("Транспонирование", "Переставляет оси тензора."),
    "Softmax": ("Softmax", "Превращает набор чисел в вероятности, сумма которых равна 1 (используется при расшифровке границ рамки)."),
    "Relu": ("ReLU", "Активация: отрицательные значения → 0."),
    "LeakyRelu": ("Leaky ReLU", "Активация: отрицательные значения слегка ослабляются, а не обнуляются."),
    "HardSwish": ("HardSwish", "Лёгкая приближённая форма SiLU (быстрее на слабом железе)."),
    "Sub": ("Вычитание", "Поэлементная разность (используется при расчёте координат рамок)."),
    "Div": ("Деление", "Поэлементное деление."),
    "MatMul": ("Матричное умножение", "Основа механизма внимания (attention): сеть сопоставляет участки картинки друг с другом, чтобы понять контекст. Встречается в блоках C2PSA / PSA современных YOLO."),
    "Gemm": ("Полносвязный слой", "Каждый выход считается из всех входов сразу."),
    "Constant": ("Константа", "Постоянное значение, зашитое в модель."),
    "Shape": ("Форма", "Служебная операция: возвращает размеры тензора."),
    "Gather": ("Выборка", "Берёт элементы по индексам."),
    "Unsqueeze": ("Добавить ось", "Служебная операция: добавляет размерность."),
    "Squeeze": ("Убрать ось", "Служебная операция: убирает размерность единичной длины."),
    "TopK": ("Лучшие K", "Выбирает K самых высоких оценок (отбор кандидатов в end-to-end моделях)."),
    "ReduceMax": ("Максимум по оси", "Находит максимальное значение (например, лучший класс)."),
    "ArgMax": ("Индекс максимума", "Номер наибольшего элемента (например, номер класса)."),
    "GlobalAveragePool": ("Глобальное усреднение", "Усредняет карту признаков до одного числа на канал."),
    "BatchNormalization": ("Нормализация", "Выравнивает значения для стабильного обучения (обычно «вшита» в Conv при экспорте)."),
    "Pad": ("Паддинг", "Добавляет рамку из нулей вокруг карты признаков."),
    "Cast": ("Смена типа", "Преобразует тип чисел (например, float32 ↔ float16)."),
    "Clip": ("Ограничение", "Обрезает значения до заданного диапазона."),
    "Sqrt": ("Корень", "Квадратный корень."),
    "Exp": ("Экспонента", "e в степени x."),
    "QuantizeLinear": ("Квантование", "Переводит значения в 8-битные целые - так модель становится меньше и быстрее."),
    "DequantizeLinear": ("Деквантование", "Возвращает 8-битные значения к обычным числам."),
    "QLinearConv": ("Свёртка (8 бит)", "Свёртка в квантованном виде."),
}


def category(op: str) -> str:
    if op in ("Conv", "QLinearConv", "ConvInteger", "Gemm", "MatMul"):
        return "conv"
    if op in ("Sigmoid", "Relu", "LeakyRelu", "HardSwish", "HardSigmoid", "Softmax", "Tanh", "Clip", "SiLU"):
        return "act"
    if op in ("MaxPool", "AveragePool", "GlobalAveragePool", "Resize", "Upsample"):
        return "pool"
    if op in ("Concat", "Split", "Slice", "Reshape", "Transpose", "Gather", "Flatten", "Squeeze", "Unsqueeze", "Pad", "Expand"):
        return "shape"
    if op in ("Add", "Mul", "Sub", "Div", "Sqrt", "Exp", "ReduceMax", "ArgMax", "TopK"):
        return "math"
    return "other"


def describe_outputs(m: OnnxModel) -> str:
    """Расшифровка формы выхода детектора YOLO простым языком."""
    if not m.outputs:
        return ""
    out = []
    for name, dt, sh in m.outputs:
        if len(sh) == 3 and all(isinstance(x, int) for x in sh[1:]):
            _, a, b = sh
            if a < b:
                nc = a - 4
                if nc > 0:
                    out.append(f"Выход «{name}» {sh}: <b>{b}</b> кандидатов, у каждого 4 числа рамки + оценки для "
                               f"<b>{nc}</b> классов (формат YOLOv8/v11/v12).")
                    continue
            if b == 6:
                out.append(f"Выход «{name}» {sh}: до <b>{a}</b> готовых рамок, у каждой x1,y1,x2,y2, уверенность, класс "
                           "(end-to-end модель без NMS: YOLOv10/26).")
                continue
            if b > 5:
                out.append(f"Выход «{name}» {sh}: <b>{a}</b> кандидатов; на каждого 4 числа рамки + уверенность объекта + "
                           f"<b>{b - 5}</b> оценок классов (формат YOLOv5/v7).")
                continue
        out.append(f"Выход «{name}»: {dt} {sh}")
    return "<br>".join(out)
