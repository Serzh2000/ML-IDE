"""Безопасная работа с сетью: только https и только известные сайты (включая все перенаправления), понятные сообщения об ошибках."""
from __future__ import annotations

import os
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request

# сайты, с которых приложение вообще что-либо скачивает (данные примеров, модели, проверка обновлений, PyTorch)
ALLOWED_HOSTS = ("github.com", "githubusercontent.com", "huggingface.co", "hf.co", "pytorch.org")


def host_ok(url: str) -> bool:
    try:
        p = urllib.parse.urlparse(url)
    except ValueError:
        return False
    host = (p.hostname or "").lower()
    if os.environ.get("MLIDE_ALLOW_LOCAL_HTTP") == "1" and host in ("127.0.0.1", "localhost") and p.scheme == "http":
        return True                                     # только для тестов: локальный сервер на этой же машине
    return p.scheme == "https" and any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not host_ok(newurl):
            raise urllib.error.URLError(f"перенаправление на неразрешённый адрес отклонено: {urllib.parse.urlparse(newurl).hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirect())


def urlopen(req, timeout: float = 60):
    url = req if isinstance(req, str) else req.full_url
    if not host_ok(url):
        raise urllib.error.URLError(f"адрес не разрешён (только https и известные сайты): {urllib.parse.urlparse(url).hostname}")
    return _opener.open(req, timeout=timeout)


def explain(exc: BaseException) -> str:
    """Понятное объяснение сетевой ошибки: прокси, сертификат, лимит запросов, нет интернета."""
    e = exc
    if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), BaseException):
        e = e.reason
    if isinstance(e, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
        return ("не удалось проверить сертификат сайта: антивирус, прокси или корпоративный фильтр подменяет защищённое соединение. "
                "Отключите проверку HTTPS в антивирусе для этой программы или добавьте корневой сертификат организации в Windows")
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 407:
            return "прокси-сервер требует авторизации (код 407): задайте переменные окружения HTTPS_PROXY с логином и паролем"
        if exc.code in (403, 429):
            return f"сайт временно ограничил число запросов (код {exc.code}): подождите несколько минут и повторите"
        return f"сайт ответил ошибкой {exc.code}"
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "сервер не отвечает (тайм-аут): проверьте интернет или повторите позже"
    if isinstance(e, (socket.gaierror, ConnectionError, OSError)):
        return "нет связи с сервером: проверьте интернет, прокси и брандмауэр"
    return f"{type(exc).__name__}: {exc}"
