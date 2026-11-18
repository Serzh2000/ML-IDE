"""Собирает сведения о наборах Roboflow-100 на Hugging Face (автор Francesco): классы, число кадров, лицензия.
Всё через публичный datasets-server, без аккаунта.  Запуск: python tools/probe_rf100.py out.json"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "mlide-catalog-builder"}


def get(url, tries=6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (i + 1))
                continue
            raise
    raise RuntimeError("rate limited")


def main():
    out = sys.argv[1]
    listing = get("https://huggingface.co/api/datasets?author=Francesco&limit=200&sort=downloads&direction=-1")
    res = []
    for x in listing:
        rid = x["id"]
        lic = next((t.split(":", 1)[1] for t in x.get("tags", []) if t.startswith("license:")), "")
        try:
            d = get(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(rid, safe='/')}&config=default&split=train&offset=0&length=1")
        except Exception as e:  # noqa
            print("skip", rid, e, file=sys.stderr)
            continue
        names = None
        for f in d.get("features", []):
            if f["name"] == "objects":
                cat = f["type"].get("category")
                inner = cat.get("feature", cat) if isinstance(cat, dict) else None
                if isinstance(inner, dict):
                    names = inner.get("names")
        row = d["rows"][0]["row"] if d.get("rows") else {}
        res.append({"id": rid, "license": lic, "classes": names or [], "train_rows": d.get("num_rows_total"),
                    "w": row.get("width"), "h": row.get("height"), "downloads": x.get("downloads")})
        print("ok", rid, d.get("num_rows_total"), len(names or []), file=sys.stderr)
        time.sleep(0.3)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("datasets:", len(res), file=sys.stderr)


if __name__ == "__main__":
    main()
