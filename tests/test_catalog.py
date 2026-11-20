"""Каталог датасетов: целостность списка, парсинг рамок, выборочная загрузка из удалённого zip (локальный HTTP-сервер с Range)."""
import http.server
import io
import os
import re
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MLIDE_ALLOW_LOCAL_HTTP"] = "1"
os.environ["MLIDE_SETTINGS_FILE"] = os.path.join(tempfile.gettempdir(), "mlide_test_catalog.ini")
from mlide import catalog as C  # noqa: E402
from mlide import paths  # noqa: E402
from mlide.model import Project  # noqa: E402

TYPES = {"zip", "zip_subset", "cls_zip", "cls_subset", "hf", "hf_rows"}


def test_catalog_integrity():
    es = C.load_catalog()
    assert len(es) >= 100, f"в каталоге должно быть около 100 наборов, а их {len(es)}"
    ids = [e["id"] for e in es]
    assert len(ids) == len(set(ids)), "id должны быть уникальны"
    for e in es:
        assert e["task"] in C.KINDS, e["id"]
        for f in ("title", "what"):
            assert C.text_of(e, f, "ru") and C.text_of(e, f, "en"), (e["id"], f)
        if not e.get("legacy"):
            ft = e["fetch"]
            assert ft["type"] in TYPES, e["id"]
            assert ft.get("url") or ft.get("repo") or ft.get("dataset"), e["id"]
            assert re.fullmatch(r"[\w\-]+", e["id"]), e["id"]
    tasks = {e["task"] for e in es}
    assert tasks == set(C.KINDS), f"нет примеров для задач: {set(C.KINDS) - tasks}"
    print("ok integrity", len(es))


def test_boxes_and_pairs():
    row = {"objects": {"bbox": [[10, 20, 30, 40], [0, 0, 0.5, 0.5]], "category": [2, 1]}}
    b = C._row_boxes(row, 100, 200)
    assert len(b) == 1 and b[0][0] == 2 and abs(b[0][1] - 0.25) < 1e-9 and abs(b[0][2] - 0.2) < 1e-9, b
    row2 = {"objects": [{"bbox": [0, 0, 50, 50], "category_id": 0}]}
    assert C._row_boxes(row2, 100, 100)[0][3] == 0.5
    names = ["a/images/train/x.jpg", "a/labels/train/x.txt", "a/images/val/y.png", "a/labels/val/y.txt", "a/images/val/z.png"]
    assert sorted(C.pair_files(names)) == [("a/images/train/x.jpg", "a/labels/train/x.txt"), ("a/images/val/y.png", "a/labels/val/y.txt")]
    print("ok boxes/pairs")


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    data = b""

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        d = type(self).data
        if rng:
            m = re.match(r"bytes=(\d+)-(\d+)", rng)
            a, b = int(m.group(1)), min(int(m.group(2)), len(d) - 1)
            body = d[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(d)}")
        else:
            body = d
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _fake_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(12):
            img = np.full((48, 64, 3), 40 + i * 5, np.uint8)
            ok, enc = cv2.imencode(".jpg", img)
            z.writestr(f"ds/images/train/img{i}.jpg", enc.tobytes())
            z.writestr(f"ds/labels/train/img{i}.txt", "1 0.5 0.5 0.4 0.4\n0 0.3 0.3 0.2 0.2\n")
        z.writestr("ds/data.yaml", "names:\n  0: cat\n  1: dog\n")
    return buf.getvalue()


def test_zip_subset():
    _RangeHandler.data = _fake_zip()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    home = Path(tempfile.mkdtemp())
    paths.examples_dir = lambda: home / "examples"
    e = {"id": "fake", "task": "detect", "title": {"ru": "t", "en": "t"}, "classes": [],
         "fetch": {"type": "zip_subset", "url": f"http://127.0.0.1:{srv.server_address[1]}/a.zip", "subset": 5}}
    p = C.build_project(e, lambda *a: True)
    pr = Project.load(p)
    assert len(pr.images) == 5, len(pr.images)
    assert pr.class_names() == ["cat", "dog"], pr.class_names()
    assert all(len(im.anns) == 2 for im in pr.images)
    srv.shutdown()
    print("ok zip_subset")


if __name__ == "__main__":
    test_catalog_integrity()
    test_boxes_and_pairs()
    test_zip_subset()
    print("ALL OK")
