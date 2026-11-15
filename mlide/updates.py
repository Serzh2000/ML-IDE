"""Проверка обновлений (только по кнопке пользователя; ничего не отправляется, кроме обычного GET к GitHub)."""
from __future__ import annotations

import json
import re
import urllib.request

REPO = "Serzh2000/ML-IDE"
API = f"https://api.github.com/repos/{REPO}/releases/latest"
PAGE = f"https://github.com/{REPO}/releases/latest"


def parse_version(s: str) -> tuple:
    nums = re.findall(r"\d+", s.split("-")[0])
    return tuple(int(x) for x in nums[:4]) or (0,)


def is_newer(current: str, latest: str) -> bool:
    a, b = parse_version(current), parse_version(latest)
    n = max(len(a), len(b))
    return b + (0,) * (n - len(b)) > a + (0,) * (n - len(a))


def fetch_latest(timeout: float = 10.0, opener=urllib.request.urlopen) -> dict:
    """Возвращает {tag, name, url, body, assets:[{name,url,size}]}. Бросает исключение при ошибке сети."""
    req = urllib.request.Request(API, headers={"User-Agent": "mlide-update-check", "Accept": "application/vnd.github+json"})
    with opener(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return {
        "tag": d.get("tag_name", ""), "name": d.get("name") or d.get("tag_name", ""), "url": d.get("html_url", PAGE),
        "body": d.get("body") or "",
        "assets": [{"name": a["name"], "url": a["browser_download_url"], "size": a.get("size", 0)} for a in d.get("assets", [])],
    }


def check(current: str, fetch=fetch_latest) -> dict:
    """{'status': 'new'|'current'|'error', 'latest': {...}, 'error': str}"""
    try:
        latest = fetch()
    except Exception as e:  # noqa
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    return {"status": "new" if is_newer(current, latest["tag"]) else "current", "latest": latest}
