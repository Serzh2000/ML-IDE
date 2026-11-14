"""Готовые учебные примеры: скачиваются по кнопке, превращаются в проект ML IDE с разметкой."""
from __future__ import annotations

import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from . import paths
from .formats import import_yolo_labels
from .model import Project

_BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
_LICENSE = "AGPL-3.0 (Ultralytics). Датасет скачивается с официального GitHub и не входит в репозиторий ML IDE."

COCO_NAMES = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
              "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
              "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
              "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
              "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
              "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
              "toothbrush").split()

EXAMPLES = [
    dict(id="wildlife", title="Дикие животные Африки", kind="Детекция", size_mb=100, url=_BASE + "african-wildlife.zip",
         names=["buffalo", "elephant", "rhino", "zebra"], subset=400, level="Лучший старт",
         what="4 класса, 400 фото (730 объектов). Данных достаточно, чтобы обучение дало отличный результат "
              "(mAP50 ≈ 0.95 на yolo11n за 30 эпох).",
         learn="Весь путь от разметки до обучения: Советник → Экспорт → Обучение → Результаты."),
    dict(id="pills", title="Таблетки (медицинские)", kind="Детекция", size_mb=8, url=_BASE + "medical-pills.zip",
         names=["pill"], subset=None, level="Быстрый",
         what="1 класс, 115 фото, около 2000 объектов (много мелких, плотно лежащих таблеток). Небольшой набор - обучение проходит быстро даже на процессоре.",
         learn="Как обучать на небольшом наборе: заморозка слоёв, аугментация, что показывают графики."),
    dict(id="signature", title="Подписи в документах", kind="Детекция", size_mb=11, url=_BASE + "signature.zip",
         names=["signature"], subset=None, level="Быстрый",
         what="1 класс, 178 фото документов с рукописными подписями разного размера: по одному объекту на кадр.",
         learn="Работа с разнородными объектами; проверка модели на вашем документе."),
    dict(id="braintumor", title="МРТ: опухоль мозга (учебный)", kind="Детекция", size_mb=4, url=_BASE + "brain-tumor.zip",
         names=["negative", "positive"], subset=None, level="Для практики",
         what="2 класса, 1116 снимков (около 590 и 575 объектов). Показывает работу со статистикой датасета и размеченными «негативными» примерами.",
         learn="Статистика датасета, проверка баланса классов, оценка результатов. Только для обучения, не для диагностики."),
    dict(id="cracks", title="Трещины (сегментация)", kind="Сегментация", size_mb=92, url=_BASE + "crack-seg.zip",
         names=["crack"], subset=300, level="Полигоны",
         what="1 класс, 300 фото, 419 контуров. Разметка - полигоны (точная форма трещины), а не рамки.",
         learn="Полигоны и «Умный контур», экспорт в формат Segmentation, обучение -seg модели."),
    dict(id="coco_small", title="COCO: 5 классов (урок про малые данные)", kind="Детекция", size_mb=7,
         url=_BASE + "coco128.zip", names=COCO_NAMES, keep_top=5, subset=None, level="Урок",
         what="128 фото из COCO, оставлены 5 самых частых классов (люди, машины, чашки…). Модель получится слабой.",
         learn="Что делают малые данные с качеством, как это видно в графиках и что советует IDE."),
]
BY_ID = {e["id"]: e for e in EXAMPLES}


class DownloadError(Exception):
    pass


def _download(url: str, dest: Path, progress):
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mlide-example-downloader"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if not progress("Загрузка", done, total):
                        raise DownloadError("Отменено")
            if zipfile.is_zipfile(dest):
                return
            last = DownloadError("Скачанный файл повреждён")
        except DownloadError:
            raise
        except Exception as e:  # noqa
            last = e
        time.sleep(2 * (attempt + 1))
    raise DownloadError(f"Не удалось скачать ({last}). Проверьте интернет и повторите.")


def _safe_extract(zip_path: Path, dest: Path):
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if dest not in target.parents and target != dest:
                raise DownloadError("Архив содержит недопустимые пути")
        z.extractall(dest)


def build_example(ex: dict, progress=lambda stage, done, total: True) -> Path:
    """Скачивает (если нужно) и собирает проект. Возвращает путь к .mlide.json."""
    root = paths.examples_dir() / ex["id"]
    proj_file = root / f"{ex['id']}.mlide.json"
    if proj_file.exists():
        return proj_file
    raw = root / "raw"
    if not raw.exists():
        zp = root / "download.zip"
        if not zp.exists():
            _download(ex["url"], zp, progress)
        progress("Распаковка", 0, 0)
        tmp = root / "raw.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        _safe_extract(zp, tmp)
        tmp.rename(raw)
        zp.unlink(missing_ok=True)

    progress("Сборка проекта", 0, 0)
    imgs = sorted(p for p in raw.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "labels" not in p.parts)
    random.Random(0).shuffle(imgs)
    if ex.get("subset"):
        imgs = imgs[:ex["subset"]]
    pr = Project(ex["title"])
    for n in ex["names"]:
        pr.add_class(n)
    pr.add_images(imgs)
    for e in pr.images:
        import_yolo_labels(e, len(ex["names"]))
        e.reviewed = True
    keep = ex.get("keep_top")
    if keep:
        counts = [sum(1 for e in pr.images for a in e.anns if a.cls == i) for i in range(len(ex["names"]))]
        top = set(sorted(range(len(counts)), key=lambda i: -counts[i])[:keep])
        for i in sorted(set(range(len(counts))) - top, reverse=True):
            pr.remove_class(i)
    pr.save(proj_file)
    return proj_file
