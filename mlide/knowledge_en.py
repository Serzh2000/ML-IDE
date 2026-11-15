"""Reference, settings advisor and a "translator" of the training process into plain language (English)."""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass

import numpy as np

# ============================================================================ glossary
GLOSSARY = [
    ("Epoch", "One full pass of the model over all training images.",
     "A model does not learn in a single pass: it sees every image many times and gradually refines itself. "
     "Typically 50-300 epochs. Too few epochs - underfitting (the model did not have time to learn), too many - "
     "overfitting (the model memorized the images). That is why people watch the charts and stop when quality on "
     "the validation data stops improving."),
    ("Batch (batch size)", "How many images the model processes in one step.",
     "A bigger batch means more stable training and faster speed on a GPU, but needs more video memory. If you get "
     "an \"out of memory\" error - reduce the batch (16 -> 8 -> 4). The value -1 enables auto-selection for your card."),
    ("Image size (imgsz)", "The size images are resized to before they are fed to the model.",
     "640 px on the long side by default. If objects in the frame are tiny (smaller than ~16 px after downscaling), "
     "they \"disappear\" - raise it to 960-1280. The price: slower training and more memory used."),
    ("Model size (n / s / m / l / x)", "How \"big\" the neural network is: nano, small, medium, large, extra-large.",
     "A bigger model is more accurate, but slower and needs more data and video memory. For a small dataset "
     "(up to ~300 images) n or s is almost always better: a large model overfits on small data. "
     "For a phone / Raspberry Pi / real time, use n."),
    ("Pretrained weights (pretrained)", "The model starts not from scratch but from weights trained on the huge COCO dataset.",
     "This is called transfer learning: the model already knows how to find edges, textures and shapes - all you "
     "need is to teach it your own classes. Thanks to this, hundreds of images are enough instead of millions. "
     "train.py does this for you."),
    ("Loss", "A number showing \"how wrong the model is\": the lower, the better.",
     "It has several parts: box_loss - accuracy of box position, cls_loss - correctness of classes, dfl_loss - "
     "fine-tuning of box edges. During training all of them should decrease smoothly. They are computed separately "
     "on the training (train) and validation (val) sets - a gap between them shows overfitting."),
    ("Precision", "What share of the found objects are really objects.",
     "Low precision = many false positives (the model \"sees\" things that are not there). Fix: add background "
     "images without objects, more varied data, raise the confidence threshold."),
    ("Recall", "What share of the objects that really exist the model found.",
     "Low recall = the model misses objects. Fix: check that you labeled ALL objects in the images "
     "(missing labels teach the model to ignore them), more data, a larger imgsz for small objects."),
    ("IoU", "How well the model's box matches yours: intersection area / union area (0..1).",
     "IoU = 1 - the boxes match perfectly. A prediction counts as correct if IoU is above a threshold (usually 0.5)."),
    ("mAP50", "The main detector metric at an IoU match threshold of 0.5 (0..1).",
     "Roughly: \"how well the model finds objects and names the class correctly\". 0.9 - excellent, "
     "0.6-0.8 - good, below 0.4 - weak. It is averaged over classes, so one bad class drags it down."),
    ("mAP50-95", "The same, but averaged over IoU thresholds from 0.5 to 0.95 - it demands precise boxes.",
     "Always lower than mAP50. If mAP50 is high but mAP50-95 is low - objects are found but the boxes are "
     "imprecise: check labeling accuracy (boxes tight around the object) and raise imgsz."),
    ("Confidence", "A number 0..1: how sure the model is about a specific detection.",
     "The confidence threshold cuts off weak detections. A high threshold - fewer false ones, but more misses. "
     "For auto-labeling it is convenient to set 0.25 and review the result."),
    ("NMS", "Duplicate removal: if the model output several boxes for one object, the best one is kept.",
     "The IoU parameter in NMS: the lower it is, the more aggressively neighboring boxes are merged. For densely "
     "packed objects raise it (0.6-0.7)."),
    ("Overfitting", "The model memorized the training images but works poorly on new ones.",
     "Sign: train loss keeps falling, while val loss stopped falling or is rising, and val mAP is flat. Fix: "
     "more data, stronger augmentation, a smaller model, early stopping (patience)."),
    ("Underfitting", "The model is too simple or trained too little - it does poorly even on train.",
     "Sign: both losses are high and still falling when training ends. Fix: more epochs, a bigger model, "
     "check the labels."),
    ("Augmentation", "Artificial variety in the data: rotations, mirroring, brightness, noise.",
     "The model sees each image slightly differently, so it does not memorize it. Augmentations must be "
     "realistic: do not flip images where left/right matters (text, arrows), do not turn street scenes "
     "upside down. For small datasets it is the cheapest way to improve quality."),
    ("Mosaic", "An online augmentation: 4 images are stitched into one.",
     "The model sees objects in different surroundings and at different scales. It is switched off automatically "
     "in the last epochs (close_mosaic) so the model can \"fine-tune\" on normal frames."),
    ("Train / Val / Test", "Three parts of a dataset: what we learn from / what we monitor by / the final exam.",
     "Train - the model learns from it. Val - the charts are built from it and the best epoch is chosen. Test - "
     "touched once at the end for an honest evaluation. Images must NOT repeat between the parts (especially "
     "frames from the same video - they are almost identical, and val will give an inflated score)."),
    ("Class imbalance", "Some objects in the labels are far more numerous than others.",
     "The model recognizes frequent classes better and rare ones worse. Rule of thumb: no more than a 10x "
     "difference. Fix: label more of the rare classes; augmentation helps only a little."),
    ("Background images", "Images without objects, with empty label files.",
     "They teach the model NOT to fire where there are no objects. 0-10% of the dataset is recommended; the "
     "images should come from the same shooting conditions where the model will work."),
    ("Detect / Segment / OBB", "Three kinds of tasks: boxes / precise outlines / rotated boxes.",
     "Detect - a rectangle around the object, the simplest and fastest. Segment - an outline mask, needed when "
     "shape/area matters. OBB - a tilted rectangle for objects at an angle (ships in aerial images, text)."),
    ("ONNX", "A universal model format that works without Python and torch.",
     "At the end train.py exports to ONNX by itself. The .onnx file can be loaded into \"Auto-label\" in this IDE "
     "or run in C++/C#/on a phone."),
    ("Early stopping (patience)", "Training stops by itself if quality has not improved for N epochs in a row.",
     "In Ultralytics patience=100 by default. For short experiments set 20-30."),
    ("Learning rate", "The size of the step by which the model corrects itself after each batch.",
     "Too large - training \"jumps\" and diverges (loss suddenly grows), too small - it learns forever. In YOLO "
     "the default is well chosen and changes on a schedule - usually you do not need to touch it."),
    ("Confusion matrix", "A table: what the model called what.",
     "It is located at runs/.../confusion_matrix.png after training. The diagonal holds correct answers. Bright "
     "spots off the diagonal show which classes get mixed up; the \"background\" row/column shows misses and "
     "false positives."),
    ("GPU and CPU", "Training on an Nvidia GPU is 10-50 times faster than on a CPU.",
     "It is possible without a GPU, but 100 epochs on a few hundred images will take hours to days. Options: "
     "Google Colab / Kaggle (free GPU) - upload the export folder and train.py there."),
]

# ============================================================================ guides
GUIDES = [
    ("Where to start (step-by-step plan)", """
<ol>
<li><b>Define the task.</b> What exactly should the model find, and in what conditions (light, angle, camera)? Shoot
images in exactly the conditions where the model will work.</li>
<li><b>Create classes</b> (on the right). Name them briefly and unambiguously; do not create extra ones "just in case".</li>
<li><b>Import photos</b> and label the first 50-100. Do not wait for perfection - auto-labeling will help later.</li>
<li><b>Check the dataset:</b> <i>Tools → Statistics and check</i> and <i>Settings advisor</i>.</li>
<li><b>Export</b> in the format you need; in the dialog click "Pick settings" - the IDE will suggest a model, epochs and a preset.</li>
<li><b>Training:</b> <i>Tools → Model training</i> - charts and explanations right in the window.</li>
<li><b>Improvement loop:</b> load the resulting best.onnx into "Auto-label", label the remaining photos with the machine,
fix the mistakes, retrain. Each round saves hours of manual work.</li>
</ol>"""),
    ("How to label correctly", """
<ul>
<li><b>Label ALL objects</b> of the needed classes in every image. The model treats a missed object as
"this is background" and learns to ignore it - the main cause of low recall.</li>
<li><b>Tight boxes</b> around the visible part of the object, without big margins, but without cutting off the edges either.</li>
<li><b>Partially hidden objects</b> - label the visible part (if that is how the model will work) - but be consistent.</li>
<li><b>Common rules</b> for disputed cases (glare, distant objects, objects cut off by the frame edge). Write them down.</li>
<li><b>Very small objects</b> (under ~8 px) are better not labeled, or increase imgsz during training.</li>
<li>Use <b>"Mark as reviewed" (R)</b> to see what has already been looked at by a human.</li>
<li><b>Always review</b> auto-label suggestions - unreviewed ones are not included in the export.</li>
</ul>"""),
    ("How much data is needed", """
<table border=1 cellpadding=4 cellspacing=0>
<tr><th>Amount</th><th>What to expect</th></tr>
<tr><td>up to 30 objects per class</td><td>Only a proof of concept; the result is unstable.</td></tr>
<tr><td>50-200 per class</td><td>A working prototype for simple tasks (one background, similar conditions).</td></tr>
<tr><td>300-1000 per class</td><td>Good quality in controlled conditions.</td></tr>
<tr><td>1500+ images, 10,000+ objects per class</td><td>The YOLOv5 authors' recommendation for maximum quality in
varied conditions.</td></tr>
</table>
<p><b>Variety</b> matters more than quantity: 100 different frames are better than 1000 nearly identical ones (frames from one video).
Different lighting, distances, backgrounds, angles. If there is little data - strong augmentation + an n/s model.</p>"""),
    ("How to read the training charts", """
<p><b>Loss</b> - should decrease. Two lines: train (training) and val (validation).</p>
<ul>
<li>Both fall and run close together - <b>all is well</b>.</li>
<li>train falls, val is flat or rising - <b>overfitting</b>: stop, take the best epoch (best.pt is saved
automatically), add data/augmentation, make the model smaller.</li>
<li>Both high and barely falling - <b>underfitting</b> / labeling problems / learning rate too small.</li>
<li>Sharp upward jumps - learning rate too large or "broken" labels (check the statistics).</li>
</ul>
<p><b>Metrics (precision, recall, mAP)</b> - should grow. For the first 3-10 epochs they are near zero - this is normal.
When mAP50 has not grown for 10-20 epochs in a row - the model has hit its ceiling on this data.</p>
<p>mAP50 guidelines: &lt;0.3 weak, 0.3-0.6 average, 0.6-0.85 good, &gt;0.85 excellent (on your own val data;
in the real world it is usually a bit worse).</p>"""),
    ("What to do if quality is low", """
<table border=1 cellpadding=4 cellspacing=0>
<tr><th>Symptom</th><th>Likely cause → action</th></tr>
<tr><td>The model misses objects</td><td>Not all objects are labeled; objects are small → raise imgsz; little data → label more.</td></tr>
<tr><td>Many false positives</td><td>Add background (images without objects); more variety; raise the confidence threshold.</td></tr>
<tr><td>Confuses similar classes</td><td>Few examples of the rare class; look at the confusion matrix; refine the labeling rules.</td></tr>
<tr><td>Good on val, bad in reality</td><td>val is too similar to train (frames from one video); the dataset has the wrong shooting conditions.</td></tr>
<tr><td>Imprecise boxes (mAP50 high, mAP50-95 low)</td><td>Sloppy labeling; increase imgsz; more epochs.</td></tr>
<tr><td>Not enough video memory</td><td>Reduce batch and/or imgsz, take a smaller model.</td></tr>
</table>"""),
    ("Deploying to ESP32-S3", """
<p>ESP32-S3 is a microcontroller without a GPU: it has about 512 KB of internal memory and up to a few MB of external memory (PSRAM).
An ordinary neural network will not fit there, so the path is special. Espressif makes the
<b>ESP-DL</b> library and the <b>ESP-PPQ</b> quantization tool for this.</p>
<ol>
<li><b>Data and labeling.</b> Shoot frames with <b>the same camera and in the same conditions</b> where the device will work
(for example, the ESP32-S3-EYE camera). As few classes as possible (1-3).</li>
<li><b>Training.</b> In "Dataset export" choose the <b>YOLO11</b> family, size <b>n</b> (the smallest),
<b>imgsz 320</b> or less. The smaller the input - the faster on the board and the lower the accuracy.</li>
<li><b>Model export.</b> <i>Tools → Model export → ESP32-S3</i> - you get an ONNX with a 320 px input.
For better quality Espressif recommends its own script <code>export_onnx.py</code> (from the esp-dl repository,
example <code>models/coco_detect</code>): it moves box decoding into post-processing and reduces losses during
quantization. Run it on your <code>best.pt</code>.</li>
<li><b>8-bit quantization.</b> ESP-PPQ turns the ONNX into a <code>.espdl</code> file. You need <b>calibration images</b> -
the "Calibration images…" button in the export window will copy ~100 of them from your project. For ESP32-S3 the Espressif
documentation gives its own quantization settings (different from ESP32-P4).</li>
<li><b>Running on the board.</b> You need <b>ESP-IDF 5.3 or newer</b>. Build the <code>yolo11_detect</code> example from esp-dl,
plugging in your <code>.espdl</code>, and flash the board.</li>
</ol>
<p><b>What to expect.</b> Quality after quantization is slightly lower than the original model; speed on ESP32-S3 is noticeably
lower than on a PC (a few frames per second, not dozens), so it suits tasks without strict real-time requirements:
counting, presence detection, simple tracking. If you need it faster - reduce the input size and the number of classes.</p>
<p><b>Alternatives.</b> Espressif is developing the <b>esp-detection</b> project with very small ESPDet-Pico models that
are trained and quantized end to end; for simple classification/FOMO tasks, Edge Impulse or TFLite Micro will do.</p>
<p style="color:#8b92a1">Documentation: Espressif ESP-DL → "How to deploy YOLO11n". Tool versions change - check the
current page.</p>"""),
    ("How to choose an export format", """
<ul>
<li><b>Ultralytics Detection</b> - the default for YOLOv5/v8/v9/v10/v11/v12/26: ordinary boxes.
The label format is the same in all these versions; only the model weights differ.</li>
<li><b>Ultralytics Segmentation</b> - if you need outlines (area, shape). Label with polygons or "Smart contour".</li>
<li><b>OBB</b> - objects lie at an arbitrary angle (aerial imagery, ships, text).</li>
<li><b>YOLOv7 / Darknet</b> - only if you work with old repositories.</li>
<li><b>COCO JSON</b> - for YOLOX, RT-DETR, Detectron2, MMDetection and most research projects.</li>
<li><b>Pascal VOC</b> - old tools and tutorials.</li>
</ul>"""),
]


def glossary_html(entry):
    t, short, long = entry
    return f"<h2>{t}</h2><p><b>{short}</b></p><p>{long}</p>"


# ============================================================================ advisor
@dataclass
class Advice:
    level: str       # ok | info | warn
    title: str
    text: str


def _labeled(project):
    return [e for e in project.images if [a for a in e.anns if not a.suggested]]


def analyze(project):
    """Returns (settings, advice). settings - keys of project.export."""
    adv: list[Advice] = []
    lab = _labeled(project)
    n = len(lab)
    names = project.class_names()
    counts = [0] * len(names)
    sides = []
    n_poly = n_box = 0
    for e in lab:
        for a in e.anns:
            if a.suggested:
                continue
            if 0 <= a.cls < len(counts):
                counts[a.cls] += 1
            x1, y1, x2, y2 = a.bbox()
            sides.append(math.sqrt(max(1e-9, (x2 - x1) * (y2 - y1))) * 640 / max(e.w, e.h))
            n_poly += a.kind == "poly"
            n_box += a.kind == "box"
    total = sum(counts)
    gpu = shutil.which("nvidia-smi") is not None
    med = float(np.median(sides)) if sides else 100.0
    small_frac = float(np.mean(np.array(sides) < 16)) if sides else 0.0

    if n == 0:
        adv.append(Advice("warn", "No labeled images", "Label at least 30-50 images and the advisor will be able to "
                          "give recommendations."))
        return {}, adv

    # --- amount of data
    if n < 30:
        adv.append(Advice("warn", f"Very little data ({n} images)",
                          "This is only enough to test an idea. Label 100+ images. Auto-labeling will help next: "
                          "train a first model on what you have and let it label the rest."))
    elif n < 200:
        adv.append(Advice("info", f"Small dataset ({n} images)",
                          "Enough for simple tasks with uniform conditions. Strong augmentation and a "
                          "small model are enabled to avoid overfitting."))
    else:
        adv.append(Advice("ok", f"Dataset size is fine ({n} images, {total} objects)",
                          "You can use a bigger model and moderate augmentation."))

    # --- classes
    few = [f"{names[i]} ({c})" for i, c in enumerate(counts) if c < 30]
    if few:
        adv.append(Advice("warn", "Few examples for some classes", "Fewer than 30 objects: " + ", ".join(few) +
                          ". The model will confuse or miss such classes - label more, up to 50-100+."))
    nz = [c for c in counts if c > 0]
    if len(nz) >= 2 and max(nz) / min(nz) > 10:
        adv.append(Advice("warn", "Class imbalance",
                          f"The most frequent class has {max(nz) / min(nz):.0f} times more objects than the rarest. "
                          "The model will get \"lazy\" on the rare ones. Label more of the rare classes."))
    empty = [names[i] for i, c in enumerate(counts) if c == 0]
    if empty:
        adv.append(Advice("info", "Classes without objects", "No labels for: " + ", ".join(empty) +
                          ". Delete them or label them - otherwise they only take up space in the model."))

    # --- object size
    imgsz = 640
    if small_frac > 0.5 or med < 20:
        imgsz = 1280 if med < 10 else 960
        adv.append(Advice("warn", "Many small objects",
                          f"The median object is ≈ {med:.0f} px at imgsz 640 ({100 * small_frac:.0f}% are smaller than 16 px). "
                          f"imgsz {imgsz} is recommended; this will slow down training. An alternative is to train on tiles."))
    elif med > 300:
        adv.append(Advice("info", "Large objects", f"Objects take up most of the frame (≈{med:.0f} px). "
                          "imgsz 640 is enough; you can even use 480 for speed."))
    else:
        adv.append(Advice("ok", "Object size suits imgsz 640", f"The median object is ≈ {med:.0f} px."))

    # --- augmentation / flips
    # NOTE: the Russian keywords are kept on purpose - they are matched against class names that users may write in Russian.
    risky = [nm for nm in names if any(k in nm.lower() for k in ("left", "right", "лев", "прав", "arrow", "стрел",
                                                                   "text", "текст", "digit", "цифр"))]
    if risky:
        adv.append(Advice("warn", "Be careful with mirroring",
                          "Class names (" + ", ".join(risky) + ") look side/direction dependent. "
                          "A horizontal flip would make them indistinguishable - in the preset set \"Horizontal "
                          "flip\" = 0."))

    # --- mixed types
    if n_poly and n_box:
        adv.append(Advice("info", "Boxes and polygons are mixed",
                          "In the Detection format polygons become enclosing boxes; in Segmentation boxes "
                          "become rectangular masks. For segmentation, label everything with polygons / \"Smart contour\"."))
    elif n_poly and not n_box:
        adv.append(Advice("info", "All labels are polygons", "For outlines choose the Segmentation format (-seg models)."))

    # --- workflow
    unrev = sum(1 for e in lab if not e.reviewed)
    sug = sum(1 for e in project.images for a in e.anns if a.suggested)
    if sug:
        adv.append(Advice("warn", f"There are unaccepted suggestions ({sug})",
                          "They will not be included in the export. Review them (the \"With suggestions\" filter) and accept with the Y key."))
    if unrev and unrev == n:
        adv.append(Advice("info", "No image is marked as reviewed",
                          "Use R to mark the ones you have looked at. In the export you can keep only reviewed ones."))
    unl = len(project.images) - n
    if unl:
        adv.append(Advice("info", f"There are unlabeled images ({unl})",
                          "If they definitely contain no objects - you can add them as background (an export option, up to ~10% of the dataset). "
                          "If there are objects - label them, otherwise the model will learn to ignore them."))
    if not gpu:
        adv.append(Advice("info", "No Nvidia GPU detected",
                          "Training on CPU is slow. For a fast result, upload the export folder to Google Colab/"
                          "Kaggle (free GPU) or stick with model n and imgsz 640."))

    # --- settings
    size = "n" if (n < 300 or not gpu) else ("s" if n < 1500 else "m")
    epochs = 200 if n < 100 else 150 if n < 500 else 100 if n < 2000 else 60
    if not gpu:
        epochs = min(epochs, 60)
        imgsz = min(imgsz, 960)
    # NOTE: the preset value MUST stay the Russian built-in preset name - presets are looked up by these keys.
    preset = "Сильная" if n < 150 else ("Средняя (по умолчанию YOLO)" if n < 1000 else "Лёгкая")
    settings = {
        "preset": preset, "epochs": epochs, "imgsz": imgsz, "size": size, "batch": 16,
        "bake_aug": n < 100,
        "train_pct": 80 if n < 300 else 70, "val_pct": 20, "test_pct": 0 if n < 300 else 10,
        "online_aug": True,
        # on small data we fine-tune gently: freeze the backbone and use a small step
        "freeze": 10 if n < 300 else 0, "lr0": 0.001 if n < 300 else 0.0, "patience": 30 if n < 300 else 100,
    }
    # display-only English alias of the preset name
    preset_en = {"Сильная": "Strong", "Средняя (по умолчанию YOLO)": "Medium (YOLO default)",
                 "Лёгкая": "Light"}.get(preset, preset)
    why = (f"Model <b>{size}</b> ({n} images{'' if gpu else ', no GPU'}), epochs <b>{epochs}</b>, imgsz <b>{imgsz}</b>, "
           f"augmentation \"<b>{preset_en}</b>\"")
    adv.insert(0, Advice("ok", "Recommended settings", why + ". The \"Apply\" button will put them into the export."))
    return settings, adv


# ============================================================================ training explanation
def _col(row, *keys):
    for k in keys:
        for kk in row:
            if kk.strip() == k:
                try:
                    return float(row[kk])
                except (TypeError, ValueError):
                    return None
    return None


def parse_rows(rows):
    """rows: list[dict] from results.csv -> dict of series."""
    S = {k: [] for k in ("epoch", "tbox", "tcls", "vbox", "vcls", "p", "r", "map50", "map")}
    for r in rows:
        ep = _col(r, "epoch")
        if ep is None:
            continue
        S["epoch"].append(ep)
        S["tbox"].append(_col(r, "train/box_loss"))
        S["tcls"].append(_col(r, "train/cls_loss"))
        S["vbox"].append(_col(r, "val/box_loss"))
        S["vcls"].append(_col(r, "val/cls_loss"))
        S["p"].append(_col(r, "metrics/precision(B)", "metrics/precision(M)"))
        S["r"].append(_col(r, "metrics/recall(B)", "metrics/recall(M)"))
        S["map50"].append(_col(r, "metrics/mAP50(B)", "metrics/mAP50(M)"))
        S["map"].append(_col(r, "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"))
    return S


def _clean(v):
    return [x for x in v if x is not None and not math.isnan(x)]


def _slope(v):
    v = _clean(v)
    if len(v) < 3:
        return 0.0
    x = np.arange(len(v))
    return float(np.polyfit(x, v, 1)[0])


def explain_training(S, total_epochs, finished=False):
    """Returns a list of (level, title, text) in plain language."""
    out = []
    ep = len(S["epoch"])
    if ep == 0:
        return [("info", "Training is starting",
                 "Preparing: checking the dataset, loading pretrained weights (the first time - a download from the "
                 "internet), building the cache. The first epoch will appear in a few minutes (longer on CPU).")]
    cur = int(S["epoch"][-1])
    total = total_epochs or max(cur, 1)
    pct = 100 * cur / total
    out.append(("info", f"Progress: epoch {cur} of {total} ({pct:.0f}%)",
                "Each epoch is one pass over all training images. After each one the model is checked on the "
                "validation (val) data - the metrics on the right are built from it."))

    m50 = _clean(S["map50"])
    p, r = _clean(S["p"]), _clean(S["r"])
    if ep < 4:
        out.append(("info", "Start: metrics are still near zero - this is normal",
                    "For the first 3-10 epochs the model \"warms up\" (warmup). Do not judge the result sooner than "
                    "10-20 epochs in."))
    if m50:
        v = m50[-1]
        band = ("weak" if v < 0.3 else "average" if v < 0.6 else "good" if v < 0.85 else "excellent")
        lvl = "warn" if v < 0.3 and ep > 15 else "ok" if v >= 0.6 else "info"
        txt = (f"mAP50 = {v:.2f} - that is \"{band}\". In simple terms: the model finds and correctly names about "
               f"{100 * v:.0f}% of the objects in the validation images (a rough estimate).")
        if v < 0.3 and ep > 15:
            txt += (" That is low for this stage: check that ALL objects are labeled, that there are enough examples per "
                    "class (Tools → Statistics) and that there are no wrong classes.")
        out.append((lvl, "Quality right now", txt))
        best_i = int(np.argmax(m50))
        since = len(m50) - 1 - best_i
        out.append(("info", "Best epoch", f"Maximum mAP50 = {m50[best_i]:.3f} at epoch {int(S['epoch'][best_i])}. "
                    "This is the version saved as best.pt - use it for your work."))
        if len(m50) >= 10 and m50[-1] < 0.7 * m50[best_i] and best_i < len(m50) - 5:
            out.append(("warn", "Quality dropped compared to the best epoch",
                        f"Now mAP50 = {m50[-1]:.2f}, while the best was {m50[best_i]:.2f} (epoch "
                        f"{int(S['epoch'][best_i])}). This usually happens on very small data: the model keeps training and "
                        "\"spoils\" what it could do. Try: more data, fewer epochs, layer freezing (freeze=10) "
                        "or a smaller learning rate (lr0=0.002)."))
        if since >= 15 and not finished:
            out.append(("warn", "Growth has stopped",
                        f"mAP has not improved for {since} epochs. The model has probably hit its ceiling on this data. "
                        "You can stop training: the risk of overfitting grows from here, and what will improve the result is "
                        "new data, not new epochs."))
        elif len(m50) >= 6 and _slope(m50[-10:]) > 0.002:
            out.append(("ok", "The model keeps improving", "Quality is growing steadily - let training continue."))

    # overfitting: val loss grows, train falls
    vb, tb = _clean(S["vbox"]), _clean(S["tbox"])
    if len(vb) >= 12 and len(tb) >= 12:
        sv, st = _slope(vb[-8:]), _slope(tb[-8:])
        vmin_i = int(np.argmin(vb))
        vmin, vlast = vb[vmin_i], vb[-1]
        if (sv > 0 and st < 0) or (vlast > 1.1 * vmin and tb[-1] < tb[0]):
            out.append(("warn", "Signs of overfitting",
                        f"The error on training images fell ({tb[0]:.2f} → {tb[-1]:.2f}), while on validation images it is worse "
                        f"than it was: minimum {vmin:.2f} (epoch {int(S['epoch'][vmin_i])}), now {vlast:.2f}. "
                        "The model is fitting itself to the training data. Take best.pt; to improve - more varied "
                        "data, stronger augmentation, a smaller model, fewer epochs."))
        elif st < 0 and sv <= 0:
            out.append(("ok", "Losses are falling on both sets", "This is a healthy picture: the model generalizes instead of memorizing."))
    elif tb and len(tb) >= 5 and _slope(tb) > 0:
        out.append(("warn", "Losses are rising", "The training error is not falling. Check the labels and the learning rate "
                    "(did you change it by hand?)."))

    # precision / recall
    if p and r and ep >= 8:
        if p[-1] - r[-1] > 0.2:
            out.append(("info", f"Precision ({p[-1]:.2f}) is higher than Recall ({r[-1]:.2f}): the model is cautious",
                        "What it finds is usually right, but it misses some objects. Check labeling completeness, "
                        "increase imgsz for small objects, add more examples."))
        elif r[-1] - p[-1] > 0.2:
            out.append(("info", f"Recall ({r[-1]:.2f}) is higher than Precision ({p[-1]:.2f}): many false positives",
                        "The model finds almost everything, but also \"sees\" extras. Add background images without objects and "
                        "more varied data; in use, raise the confidence threshold."))
        else:
            out.append(("ok", f"Precision {p[-1]:.2f} and Recall {r[-1]:.2f} are balanced",
                        "The model tends neither to miss objects nor to invent extra ones."))

    if finished:
        out.append(("ok", "Training finished",
                    "Results are in the runs/… folder: weights/best.pt (the best model), results.png (charts), "
                    "confusion_matrix.png (which classes get mixed up). train.py has also made best.onnx - load it into "
                    "\"Auto-label\" so the model labels the remaining images, fix the mistakes and retrain."))
    return out
