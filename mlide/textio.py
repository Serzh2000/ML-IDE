"""Чтение текстовых файлов с разными кодировками (Блокнот и PowerShell пишут BOM, русские файлы бывают в cp1251)."""
from __future__ import annotations

import codecs
from pathlib import Path


def decode_bytes(raw: bytes, truncated: bool = False) -> str:
    """utf-8 (с BOM или без) -> cp1251 -> latin-1. UTF-16 только при явном BOM (иначе чётная длина cp1251 даёт «китайский» мусор).
    truncated=True: данные оборваны посреди файла - неполный символ в конце отбрасывается."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16")
        except UnicodeError:
            pass
    try:
        if truncated:
            return codecs.getincrementaldecoder("utf-8-sig")("strict").decode(raw, final=False)
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1251")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_text(path, limit: int | None = None) -> str:
    p = Path(path)
    if limit is None:
        return decode_bytes(p.read_bytes())
    with open(p, "rb") as f:
        raw = f.read(limit + 1)
    return decode_bytes(raw[:limit], truncated=len(raw) > limit)


def read_lines(path) -> list[str]:
    return read_text(path).splitlines()


def decode_stream(owner, data, attr: str = "_utf8_dec") -> str:
    """Вывод процесса приходит кусками по произвольным границам: многобайтный символ (кириллица, полоски прогресса) нельзя
    декодировать по частям - состояние декодера хранится на процессе-владельце."""
    dec = getattr(owner, attr, None)
    if dec is None:
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        setattr(owner, attr, dec)
    return dec.decode(bytes(data))
