"""Просмотр модели через Netron: локальный сервер отдаёт страницу, модель и не выпускает файлы за пределы assets/netron."""
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlide import netron_view as N


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read(), r.headers


def main():
    assert N.available(), "assets/netron не найден"
    tmp = Path(tempfile.mkdtemp()) / "кириллица"
    tmp.mkdir()
    model = tmp / "best.onnx"
    model.write_bytes(b"\x08\x07fake-model")
    url = N.url_for(str(model))
    base = url.split("/?")[0]
    st, body, hd = get(base + "/")
    html = body.decode("utf-8")
    assert st == 200 and 'name="version" content="0.0.0"' in html, "статистика Netron должна быть выключена"
    assert "consent=1" in hd.get("Set-Cookie", ""), "cookie согласия должен выставляться (иначе Netron обращается к ipinfo.io)"
    st, body, _ = get(base + "/index.js")
    assert st == 200 and len(body) > 100
    mpath = urllib.parse.unquote(url.split("?url=")[1])
    st, body, hd2 = get(base + mpath)
    assert body == model.read_bytes(), "модель должна отдаваться как есть"
    assert "connect-src 'self'" in hd2.get("Content-Security-Policy", ""), "нужен CSP без внешних подключений"
    other = Path(tempfile.mkdtemp()) / "best.onnx"                       # второй файл с тем же именем: свой адрес
    other.write_bytes(b"other")
    url2 = N.url_for(str(other))
    assert url2 != url and get(base + urllib.parse.unquote(url2.split("?url=")[1]))[1] == b"other"
    assert get(base + mpath)[1] == model.read_bytes(), "первая модель не должна быть перезаписана второй"
    req = urllib.request.Request(base + mpath, headers={"Host": "evil.example.com"})        # DNS-rebinding: чужой Host
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("запрос с чужим Host должен отклоняться")
    except urllib.error.HTTPError as e:
        assert e.code == 403
    for bad in ("/../mlide/__init__.py", "/..%2f..%2fREADME.md", "/model/nope.onnx", "/model/x/y.onnx", "/nope.js"):
        try:
            get(base + bad)
            raise AssertionError(f"{bad} не должен отдаваться")
        except urllib.error.HTTPError as e:
            assert e.code == 404, (bad, e.code)
    print("ok netron")
    print("ALL OK")


if __name__ == "__main__":
    main()
