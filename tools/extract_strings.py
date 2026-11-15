"""Собирает все русские строки интерфейса из кода: python tools/extract_strings.py [out.json]

Результат - список уникальных фрагментов (целые строки и постоянные части f-строк). По нему составляется
словарь перевода assets/i18n_en.json (см. mlide/i18n.py).
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "mlide"
SKIP = {"knowledge.py", "knowledge_en.py", "i18n.py"}
CYR = re.compile(r"[А-Яа-яЁё]")


def docstring_nodes(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                ids.add(id(body[0].value))
    return ids


def collect():
    found: dict[str, str] = {}
    for p in sorted(ROOT.rglob("*.py")):
        if p.name in SKIP:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        docs = docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if CYR.search(node.value):
                    found.setdefault(node.value, p.name)
    return found


if __name__ == "__main__":
    strings = collect()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("strings_ru.json")
    out.write_text(json.dumps(sorted(strings), ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(strings)} уникальных русских фрагментов, {sum(map(len, strings))} символов -> {out}")
