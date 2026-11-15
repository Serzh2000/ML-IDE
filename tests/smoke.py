"""Headless smoke test: python tests/smoke.py"""
import os, sys, tempfile, json
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MLIDE_SETTINGS_FILE", os.path.join(__import__("tempfile").gettempdir(), "mlide_test_settings.ini"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, cv2

from mlide import imgio, autolabel
from mlide.model import Project, Annotation
from mlide.formats import FORMATS, ExportOptions, export_dataset, import_yolo_labels
from mlide.augment import BUILTIN_PRESETS, augment_sample

tmp = Path(tempfile.mkdtemp(prefix="mlide_"))
imgs = tmp / "фото"; imgs.mkdir()            # non-ASCII path on purpose
rng = np.random.default_rng(1)
for i in range(10):
    im = np.full((300, 400, 3), 60, np.uint8)
    cv2.rectangle(im, (100, 80), (220, 200), (0, 200, 255), -1)
    imgio.imwrite(imgs / f"img{i}.jpg", im)

pr = Project("t"); pr.add_class("thing"); pr.add_class("other")
assert pr.add_images(sorted(imgs.glob("*.jpg"))) == 10
for e in pr.images[:8]:
    e.anns.append(Annotation(0, "box", [[100, 80], [220, 200]]))
    e.anns.append(Annotation(1, "poly", [[10, 10], [60, 10], [40, 60]]))
pr.save(tmp / "p.mlide.json")
pr2 = Project.load(tmp / "p.mlide.json")
assert len(pr2.images) == 10 and len(pr2.images[0].anns) == 2

for k in FORMATS:
    if k in ("ultra_pose", "ultra_cls"):          # нужны ключевые точки / классы кадров: см. tests/test_new_tasks.py
        continue
    out = tmp / ("out_" + k)
    o = ExportOptions(fmt=k, out_dir=str(out), bake_aug=True, aug=BUILTIN_PRESETS["Сильная"], seed=3)
    print(k, "->", export_dataset(pr2, o).splitlines()[0])
    assert any(out.rglob("*")), k
y = (tmp / "out_ultra_det" / "data.yaml").read_text(); print(y)
assert (tmp / "out_ultra_det" / "train.py").exists()
print((next((tmp / "out_ultra_det" / "labels" / "train").glob("*.txt"))).read_text())

# round trip of yolo import
e = pr2.images[9]; e.anns.clear()
lbl = next((tmp / "out_ultra_det" / "labels" / "train").glob("img*_aug0.txt"), None)

# augmentation sanity: boxes still on the object after hflip-only
from mlide.augment import AugParams
p = AugParams(fliplr=1, flipud=0, degrees=0, translate=0, scale=0, hsv_h=0, hsv_s=0, hsv_v=0)
im = imgio.imread(pr2.images[0].path)
o, a = augment_sample(im, [(0, "box", [[100, 80], [220, 200]])], p, rng)
assert a[0][2] == [[180.0, 80.0], [300.0, 200.0]], a
print("aug flip ok")

# smart polygon
poly = autolabel.smart_polygon(im, (90, 70, 235, 215)); print("smart poly pts:", len(poly or []))
assert poly

# UI
from PySide6.QtWidgets import QApplication
from mlide.ui.style import apply_dark
from mlide.ui.main_window import MainWindow
from mlide.ui.dialogs import PresetDialog, ExportDialog, AutoLabelDialog, StatsDialog
app = QApplication([]); apply_dark(app)
w = MainWindow(); w.show()
w._open_path(str(tmp / "p.mlide.json"))
w.set_tool("select"); w.canvas.select(w.entry.anns[0].id)
w.pick_class(1); w.undo(); w.redo(); w.step(1); w.copy_prev()
w.resize(1300, 800); w.grab().save(str(tmp / "shot.png"))
PresetDialog(w.project, w.entry, w); ExportDialog(w.project, w); AutoLabelDialog(w.project, w.entry, w); StatsDialog(w.project, w)
print("UI ok; screenshot:", tmp / "shot.png")

# ---- knowledge / advisor / training explanation
from mlide import knowledge as K
settings, adv = K.analyze(pr2)
print("advisor:", settings, [a.title for a in adv][:4])
assert settings and adv
import csv, io
rows = []
for ep in range(1, 41):
    rows.append({" epoch": ep, "train/box_loss": 2 - ep * 0.04, "val/box_loss": 2 - ep * 0.03 + (0.05 * (ep - 25) if ep > 25 else 0),
                 "metrics/precision(B)": min(0.9, ep * 0.03), "metrics/recall(B)": min(0.5, ep * 0.02),
                 "metrics/mAP50(B)": min(0.7, ep * 0.03) if ep < 25 else 0.7 - 0.001 * ep, "metrics/mAP50-95(B)": 0.3})
S = K.parse_rows(rows)
for lvl, t, _ in K.explain_training(S, 100): print(" ", lvl, t)
assert K.explain_training(K.parse_rows([]), 50)
from mlide.ui.learn import HelpDialog, AdvisorDialog, TrainDialog, LineChart
from mlide.ui.main_window import SHORTCUTS_HTML
HelpDialog(SHORTCUTS_HTML, w, "Как читать"); AdvisorDialog(w.project, w)
td = TrainDialog(w.project, w); td.c_loss.set_data(S["epoch"], {"tbox": S["tbox"], "vbox": S["vbox"]})
td.explain.setHtml(__import__("mlide.ui.learn", fromlist=["x"]).cards_html(K.explain_training(S, 100)))
td.grab().save(str(tmp / "train.png")); print("train dialog:", tmp / "train.png")

# ---- 0.2: стартовый экран, примеры, результаты, тур
from mlide.ui.examples_dialog import ExamplesDialog
from mlide.ui.results import ResultsDialog
ExamplesDialog(w); ResultsDialog(w.project, w)
w2 = MainWindow(); assert w2.stack.currentIndex() == 0          # пустой проект -> стартовый экран
w2.start_tour(); assert len(w2._tour.steps) >= 10; w2._tour.finish()
print("0.2 UI ok")

# ---- 0.3: разбор ONNX на синтетической модели, экспорт модели, просмотр
def _v(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)
def _f(num, wt, payload):
    return _v((num << 3) | wt) + payload
def _len(num, b):
    return _f(num, 2, _v(len(b)) + b)
def _str(num, s):
    return _len(num, s.encode())
def _vi(num, n):
    return _f(num, 0, _v(n))
def _value_info(name, dims):
    shape = b"".join(_len(1, _vi(1, d)) for d in dims)
    tensor = _vi(1, 1) + _len(2, shape)
    return _str(1, name) + _len(2, _len(1, tensor))
conv = _str(1, "x") + _str(1, "w") + _str(2, "y") + _str(3, "conv0") + _str(4, "Conv") + \
    _len(5, _str(1, "kernel_shape") + _vi(8, 3) + _vi(8, 3) + _vi(20, 7))
relu = _str(1, "y") + _str(2, "z") + _str(3, "relu0") + _str(4, "Relu")
weight = _vi(1, 4) + _vi(1, 3) + _vi(1, 3) + _vi(1, 3) + _vi(2, 1) + _str(8, "w")
graph = _len(1, conv) + _len(1, relu) + _str(2, "g") + _len(5, weight) + _len(11, _value_info("x", [1, 3, 8, 8])) + \
    _len(12, _value_info("z", [1, 4, 6, 6]))
model_bytes = _vi(1, 8) + _str(2, "test") + _len(7, graph) + _len(8, _str(1, "") + _vi(2, 18))
onnx_path = tmp / "tiny.onnx"
onnx_path.write_bytes(model_bytes)
from mlide import onnxparse, modelexport
om = onnxparse.load_onnx(onnx_path)
assert [n.op for n in om.nodes] == ["Conv", "Relu"] and om.params == 4 * 3 * 3 * 3, (om.nodes, om.params)
assert om.inputs[0][2] == [1, 3, 8, 8] and om.nodes[0].attrs["kernel_shape"] == [3, 3]
cmd = modelexport.build_command("python", "best.pt", "esp32", 320)
assert '"format": "onnx"' in cmd[-1] and '"imgsz": 320' in cmd[-1], cmd[-1]
assert modelexport.build_command("python", "b.pt", "onnx", half=True)[-1].count('"half": true') == 1
from mlide.ui.model_viewer import ModelViewerDialog
from mlide.ui.model_export import ModelExportDialog
vd = ModelViewerDialog(w, str(onnx_path)); assert len(vd.items) == 4      # вход + 2 слоя + выход
ModelExportDialog(w.project, w)
print("0.3 UI/model ok")

# ---- 0.4: новые окна и логика (русский интерфейс)
from mlide import classtools, importers, dedupe, video, sam as sam_mod
from mlide.ui.video_dialog import VideoDialog
from mlide.ui.dedupe_dialog import DedupeDialog
from mlide.ui.review_dialog import ReviewDialog, analyze_labels
from mlide.ui.compare_dialog import CompareDialog
from mlide.ui.live_test import LiveTestDialog
from mlide.ui.help_dialogs import GpuHelpDialog, CodeSnippetsDialog
from mlide.ui.imagelist import ImageListModel
for dlg in (VideoDialog(w), DedupeDialog(w.project, w), ReviewDialog(w.project, w), CompareDialog(w.project, w),
            LiveTestDialog(w.project, w), GpuHelpDialog(w), CodeSnippetsDialog("best.onnx", ["a", "b"], 640, w)):
    assert dlg is not None
from mlide.autolabel import Detection
sc, issues = analyze_labels([(0, (10, 10, 100, 100))], [Detection(1, 0.9, (12, 12, 98, 98))], ["a", "b"])
assert issues and sc > 1.0
m = ImageListModel(lambda: w.project); m.rebuild(0); assert m.rowCount() == len(w.project.images)
m.rebuild(1); assert all(not [a for a in w.project.images[m.image_index(r)].anns if not a.suggested] for r in range(m.rowCount()))
assert not sam_mod.models_ready() or sam_mod.runtime_available()
# английские фрагменты знаний подключаются переключателем языка
import subprocess
r = subprocess.run([sys.executable, "-c", "import mlide.knowledge as K; print(K.GUIDES[0][0])"], capture_output=True, text=True,
                   encoding="utf-8", env={**os.environ, "MLIDE_LANG": "en", "PYTHONIOENCODING": "utf-8"})
assert "Where to start" in r.stdout, r.stdout + r.stderr
print("0.4 UI/model ok")
