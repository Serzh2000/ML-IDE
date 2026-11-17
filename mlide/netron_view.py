"""Просмотр модели в Netron (https://github.com/lutzroeder/netron, MIT).

Netron - это веб-приложение, поэтому оно лежит в assets/netron и открывается в браузере пользователя через маленький
локальный сервер (только 127.0.0.1). Модель никуда не отправляется: сервер отдаёт файл самому себе. Статистику и
cookie-баннер Netron отключаем (подменяем версию на 0.0.0 и выставляем cookie согласия, чтобы он не обращался к сети)."""
from __future__ import annotations

import http.server
import mimetypes
import re
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from . import paths

_MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".png": "image/png", ".ico": "image/x-icon", ".svg": "image/svg+xml"}

_state: dict = {"server": None, "files": {}, "lock": threading.Lock()}


def root() -> Path:
    return paths.resource("assets/netron")


def available() -> bool:
    return (root() / "index.html").exists()


def _index_html() -> bytes:
    s = (root() / "index.html").read_text(encoding="utf-8")
    s = re.sub(r'(<meta name="version" content=")[^"]*(")', r"\g<1>0.0.0\g<2>", s)            # без статистики и проверки версии
    s = re.sub(r'(<meta name="date" content=")[^"]*(")', r"\g<1>" + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + r"\g<2>", s)
    return s.encode("utf-8")


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):                      # тишина в консоли
        pass

    def _send(self, code, body=b"", ctype="text/plain", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path in ("/", "/index.html"):
            cookie = "consent=1; Path=/; Max-Age=31536000; SameSite=Strict"
            return self._send(200, _index_html(), _MIME[".html"], {"Set-Cookie": cookie})
        if path.startswith("/model/"):
            f = _state["files"].get(path[len("/model/"):])
            if f is None or not Path(f).is_file():
                return self._send(404, b"not found")
            return self._send(200, Path(f).read_bytes(), "application/octet-stream")
        rel = path.lstrip("/")
        base = root().resolve()
        target = (base / rel).resolve()
        if base not in target.parents or not target.is_file():
            return self._send(404, b"not found")
        ctype = _MIME.get(target.suffix.lower()) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return self._send(200, target.read_bytes(), ctype)


def _ensure_server():
    with _state["lock"]:
        if _state["server"] is None:
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            _state["server"] = srv
        return _state["server"]


def url_for(model_path: str) -> str:
    """Адрес страницы Netron с загруженной моделью (запускает сервер при необходимости)."""
    p = Path(model_path)
    srv = _ensure_server()
    name = re.sub(r"[^\w.\-]+", "_", p.name, flags=re.A) or "model"
    _state["files"][name] = str(p)
    port = srv.server_address[1]
    return f"http://127.0.0.1:{port}/?url={urllib.parse.quote('/model/' + name)}"


def open_in_browser(model_path: str) -> str:
    """Открывает модель в Netron в браузере по умолчанию. Возвращает адрес."""
    url = url_for(model_path)
    webbrowser.open(url)
    return url
