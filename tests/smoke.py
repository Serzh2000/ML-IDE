"""Headless smoke test: python tests/smoke.py"""
import os, sys, tempfile, json
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
