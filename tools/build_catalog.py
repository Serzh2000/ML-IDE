"""Собирает assets/datasets_catalog.json из: списка Ultralytics (ниже), Roboflow-100 на Hugging Face (tools/data/rf100_*.json)
и нескольких отдельных репозиториев Hugging Face.  Запуск: python tools/build_catalog.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mlide.examples import COCO_NAMES  # noqa: E402

B = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
LIC_UL = "AGPL-3.0 (Ultralytics)"
LIC_RF = "CC BY 4.0 (Roboflow 100)"


def e(id, task, ru, en, wru, wen, lru, len_, domain, size, fetch, lic=LIC_UL, classes=None, source="ultralytics", **kw):
    d = {"id": id, "task": task, "title": {"ru": ru, "en": en}, "what": {"ru": wru, "en": wen}, "learn": {"ru": lru, "en": len_},
         "domain": domain, "license": lic, "size_mb": size, "classes": classes or [], "source": source, "fetch": fetch}
    d.update(kw)
    return d


UL = [
    e("coco8", "detect", "COCO8: 8 картинок", "COCO8: 8 pictures", "8 фото из COCO, 80 классов: самый маленький набор для проверки всего пути за минуту.",
      "8 COCO photos, 80 classes: the smallest set for trying the whole workflow in a minute.", "Быстрая проверка экспорта и обучения.",
      "A quick check of export and training.", "everyday", 0.4, dict(type="zip", url=B + "coco8.zip"), classes=COCO_NAMES),
    e("coco32", "detect", "COCO32: 32 картинки", "COCO32: 32 pictures", "32 фото из COCO с разметкой людей, машин, предметов быта.",
      "32 COCO photos with people, vehicles and household objects labeled.", "Мало данных: как это видно в графиках.",
      "Little data: how it shows up in the charts.", "everyday", 2, dict(type="zip", url=B + "coco32.zip"), classes=COCO_NAMES),
    e("coco128", "detect", "COCO128: 128 картинок", "COCO128: 128 pictures", "128 фото из COCO, 80 классов - классический учебный набор Ultralytics.",
      "128 COCO photos, 80 classes - the classic Ultralytics learning set.", "Много классов и мало примеров на каждый.",
      "Many classes with few examples each.", "everyday", 7, dict(type="zip", url=B + "coco128.zip"), classes=COCO_NAMES),
    e("homeobjects3k", "detect", "Предметы дома (3000 фото)", "Home objects (3000 photos)",
      "Фото интерьеров: кровать, диван, стул, раковина и другие предметы (12 классов).",
      "Interior photos: bed, sofa, chair, sink and other items (12 classes).", "Реалистичные сцены с перекрытием предметов.",
      "Realistic scenes with overlapping objects.", "everyday", 120, dict(type="zip_subset", url=B + "homeobjects-3K.zip", subset=300)),
    e("construction_ppe", "detect", "Стройка: каски и жилеты", "Construction safety gear", "Рабочие на стройке: каски, жилеты, перчатки, очки и нарушения (12 классов).",
      "Workers on a construction site: helmets, vests, gloves, goggles and violations (12 classes).", "Задача безопасности: классы «есть» и «нет» средства защиты.",
      "A safety task: classes for 'has' and 'lacks' protective gear.", "industrial", 90, dict(type="zip_subset", url=B + "construction-ppe.zip", subset=300)),
    e("kitti", "detect", "KITTI: дорожные сцены", "KITTI: road scenes", "Снимки с камеры автомобиля: машины, фургоны, пешеходы, велосипедисты (8 классов).",
      "Dashcam-style photos: cars, vans, pedestrians, cyclists (8 classes).", "Вытянутые кадры и много мелких объектов.",
      "Wide frames with many small objects.", "everyday", 120, dict(type="zip_subset", url=B + "kitti.zip", subset=300)),
    e("coco8_seg", "seg", "COCO8-seg: контуры", "COCO8-seg: outlines", "8 фото из COCO, объекты размечены полигонами (80 классов).",
      "8 COCO photos with objects labeled as polygons (80 classes).", "Минимальный пример сегментации.", "A minimal segmentation example.",
      "everyday", 0.4, dict(type="zip", url=B + "coco8-seg.zip"), classes=COCO_NAMES),
    e("coco128_seg", "seg", "COCO128-seg: контуры", "COCO128-seg: outlines", "128 фото из COCO с контурами объектов (80 классов).",
      "128 COCO photos with object outlines (80 classes).", "Сегментация на небольшом наборе.", "Segmentation on a small set.",
      "everyday", 7, dict(type="zip", url=B + "coco128-seg.zip"), classes=COCO_NAMES),
    e("carparts_seg", "seg", "Детали автомобиля (сегментация)", "Car parts (segmentation)", "Фото автомобилей: бамперы, двери, стёкла, фары и другие детали (23 класса) контурами.",
      "Car photos: bumpers, doors, glass, lights and other parts (23 classes) as outlines.", "Много классов-деталей одного объекта.",
      "Many part classes of one object.", "industrial", 40, dict(type="zip_subset", url=B + "carparts-seg.zip", subset=300)),
    e("package_seg", "seg", "Посылки (сегментация)", "Packages (segmentation)", "Фото коробок и посылок с контурами.",
      "Photos of boxes and parcels with outlines.", "Один класс сложной формы на разном фоне.", "One class of varied shape on varied backgrounds.",
      "industrial", 40, dict(type="zip_subset", url=B + "package-seg.zip", subset=300)),
    e("tiger_pose", "pose", "Тигры: ключевые точки", "Tigers: keypoints", "Фото тигров с 12 ключевыми точками на теле.",
      "Tiger photos with 12 keypoints on the body.", "Ключевые точки у животных.", "Keypoints on animals.",
      "nature", 25, dict(type="zip_subset", url=B + "tiger-pose.zip", subset=263)),
    e("hand_keypoints", "pose", "Руки: 21 точка", "Hands: 21 keypoints", "Фото рук с 21 точкой на каждой (суставы пальцев и запястье).",
      "Hand photos with 21 keypoints on each (finger joints and wrist).", "Много точек на небольшом объекте.", "Many points on a small object.",
      "everyday", 60, dict(type="zip_subset", url=B + "hand-keypoints.zip", subset=300)),
    e("dog_pose", "pose", "Собаки: ключевые точки", "Dogs: keypoints", "Фото собак с 24 ключевыми точками на теле.",
      "Dog photos with 24 keypoints on the body.", "Ключевые точки у животных разных пород и поз.", "Keypoints on animals of different breeds and poses.",
      "nature", 60, dict(type="zip_subset", url=B + "dog-pose.zip", subset=300)),
    e("dota128", "obb", "Аэроснимки DOTA128", "Aerial DOTA128", "128 аэроснимков, объекты размечены повёрнутыми рамками (самолёты, корабли, площадки).",
      "128 aerial images with objects labeled as rotated boxes (planes, ships, courts).", "Повёрнутые рамки (OBB) на аэроснимках.", "Rotated boxes (OBB) on aerial images.",
      "aerial", 33, dict(type="zip", url=B + "dota128.zip"),
      classes=["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge",
               "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"]),
    e("mnist2560", "cls", "Цифры MNIST (2560)", "MNIST digits (2560)", "2560 картинок рукописных цифр 0-9: классом помечен весь кадр.",
      "2560 handwritten digit images 0-9: the whole frame gets the class.", "Классификация: быстрый и лёгкий пример.", "Classification: a fast, light example.",
      "everyday", 1.1, dict(type="cls_zip", url=B + "mnist2560.zip", per_class=40)),
    e("imagenet10", "cls", "ImageNet-10", "ImageNet-10", "Мини-подборка ImageNet: 10 классов обычных предметов и животных.",
      "A mini ImageNet subset: 10 classes of everyday objects and animals.", "Классификация фотографий реального мира.", "Classifying real-world photos.",
      "everyday", 0.1, dict(type="cls_zip", url=B + "imagenet10.zip", per_class=40)),
    e("imagenet100", "cls", "ImageNet-100", "ImageNet-100", "Подборка ImageNet на 100 классов.",
      "An ImageNet subset with 100 classes.", "Много классов: на что обращать внимание в Советнике.", "Many classes: what to watch for in the Advisor.",
      "everyday", 0.7, dict(type="cls_zip", url=B + "imagenet100.zip", per_class=10)),
    e("fashion_mnist", "cls", "Fashion-MNIST: одежда", "Fashion-MNIST: clothing", "Чёрно-белые картинки одежды и обуви (10 классов).",
      "Grayscale pictures of clothing and shoes (10 classes).", "Классификация похожих между собой классов.", "Classifying look-alike classes.",
      "everyday", 49, dict(type="cls_zip", url=B + "fashion-mnist.zip", per_class=40)),
    e("imagenette160", "cls", "Imagenette: 10 лёгких классов", "Imagenette: 10 easy classes", "10 хорошо различимых классов ImageNet: рыба, собака, здание, церковь, самолёт и др.",
      "10 easily distinguishable ImageNet classes: fish, dog, building, church, plane and more.", "Классификация фото: быстрый старт.", "Photo classification: a quick start.",
      "everyday", 107, dict(type="cls_zip", url=B + "imagenette160.zip", per_class=40)),
    e("imagewoof160", "cls", "Imagewoof: 10 пород собак", "Imagewoof: 10 dog breeds", "10 похожих друг на друга пород собак: трудная классификация.",
      "10 similar dog breeds: a hard classification task.", "Тонкие различия между классами.", "Subtle differences between classes.",
      "nature", 100, dict(type="cls_zip", url=B + "imagewoof160.zip", per_class=40)),
    e("caltech101", "cls", "Caltech-101", "Caltech-101", "Фото предметов из 101 категории (самолёты, стулья, лица, мотоциклы...).",
      "Photos of objects from 101 categories (planes, chairs, faces, motorbikes...).", "Много категорий, неравное число картинок.",
      "Many categories with unequal counts.", "everyday", 137, dict(type="cls_zip", url=B + "caltech101.zip", per_class=10)),
]

HF_RAW = [
    e("hf_obstacle", "detect", "Препятствия на пути (велосипед, человек, машина)", "Obstacles on the road", "Снимки дороги: велосипеды, здания, машины, люди и другие препятствия.",
      "Road photos: bicycles, buildings, cars, people and other obstacles.", "Крупный набор: берём 300 случайных кадров.", "A large set: 300 random frames are used.",
      "everyday", 40, dict(type="hf", repo="Abtinzandi/Obstacle-Detection-Dataset-YOLO", subset=300), lic="MIT", source="huggingface"),
    e("hf_soccer", "detect", "Футбол: мяч и игроки", "Football: ball and players", "Кадры футбольных матчей с роботами и мячом (NUbots).",
      "Football match frames with robots and a ball (NUbots).", "Мелкие быстрые объекты.", "Small fast objects.",
      "game", 60, dict(type="hf", repo="NUbots/YOLOSoccer", subset=300), lic="MIT", source="huggingface"),
    e("hf_pothole", "detect", "Ямы на дороге", "Potholes", "Фото дорожного покрытия: ямы отмечены рамками.",
      "Road surface photos with potholes marked by boxes.", "Один класс на реальных фото дорог.", "One class on real road photos.",
      "everyday", 22, dict(type="hf", repo="Ryukijano/Pothole-detection-Yolov8", subset=300), lic="OpenRAIL", source="huggingface", classes=["pothole"]),
    e("hf_basketball", "pose", "Баскетбольная площадка: 48 точек", "Basketball court: 48 keypoints", "Кадры с размеченной площадкой: 48 ключевых точек разметки.",
      "Frames with the court marked by 48 keypoints.", "Много точек на одной сцене.", "Many points on one scene.",
      "everyday", 80, dict(type="hf", repo="koppolusameer/basketball-keypoint-detection-yolov8", subset=300), lic="MIT", source="huggingface"),
]


def rf100():
    probe = json.loads((ROOT / "tools/data/rf100_probe.json").read_text(encoding="utf-8"))
    texts = json.loads((ROOT / "tools/data/rf100_texts.json").read_text(encoding="utf-8"))
    out = []
    for x in probe:
        t = texts.get(x["slug"])
        if not t:
            continue
        n = min(int(x["train_rows"] or 0), 300)
        kb = max(25, (x.get("w") or 640) * (x.get("h") or 640) / (640 * 640) * 60)
        out.append({"id": "rf_" + x["slug"].replace("-", "_"), "task": "detect",
                    "title": {"ru": t["title_ru"], "en": t["title_en"]}, "what": {"ru": t["what_ru"], "en": t["what_en"]},
                    "learn": {"ru": t["learn_ru"], "en": t["learn_en"]}, "domain": t["domain"], "license": LIC_RF,
                    "size_mb": max(1, round(n * kb / 1024)), "images": n, "classes": x["classes"], "source": "huggingface",
                    "fetch": {"type": "hf_rows", "dataset": x["id"], "subset": n}})
    return out


def main():
    entries = UL + HF_RAW + rf100()
    ids = [x["id"] for x in entries]
    assert len(ids) == len(set(ids)), "дубли id"
    out = {"version": 1, "entries": entries}
    (ROOT / "assets" / "datasets_catalog.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(entries), "записей")


if __name__ == "__main__":
    main()
