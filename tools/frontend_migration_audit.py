#!/usr/bin/env python3
"""Audit that the frontend migration stays on the C++ parser/emitter path."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REMOVED_FRONTEND_PATHS = [
    ROOT / "src/lython/frontend",
    ROOT / "src/lython/visitors",
]

ALLOWED_PYTHON_SOURCES = {
    ROOT / "src/lython/__init__.py",
}

ALLOWED_AST_PARSE_REFERENCES = {
    ROOT / "tools/frontend_migration_audit.py",
    ROOT / "tools/CLI.cpp",
    ROOT / "tools/parser_ast_parity_smoke.py",
    ROOT / "src/lython/parser/README.md",
    ROOT / "src/lython/emitter/README.md",
}

CPYTHON_SPEC_TEXT_FILES = {
    ROOT / "src/lython/parser/parser.c",
}
PRUNED_TEXT_DIRS = {
    ROOT / ".git",
    ROOT / ".venv",
    ROOT / "build",
    ROOT / "third_party",
}


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_text_files() -> list[Path]:
    result: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if any(is_under(path, pruned) for pruned in PRUNED_TEXT_DIRS):
            continue
        result.append(path)
    return result


def check_removed_paths(failures: list[str]) -> None:
    for path in REMOVED_FRONTEND_PATHS:
        if path.exists():
            failures.append(f"removed Python frontend path still exists: {path.relative_to(ROOT)}")


def check_python_sources(failures: list[str]) -> None:
    for path in (ROOT / "src/lython").rglob("*.py"):
        if path in ALLOWED_PYTHON_SOURCES:
            continue
        failures.append(f"unexpected Python source under src/lython: {path.relative_to(ROOT)}")


def check_legacy_references(failures: list[str]) -> None:
    needles = ["lython.frontend", "lython.visitors"]
    for path in iter_text_files():
        if path == ROOT / "tools/frontend_migration_audit.py":
            continue
        if path.suffix not in {".py", ".cpp", ".h", ".md", ".txt", ".yml", ".toml"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for needle in needles:
            if needle in text:
                failures.append(f"legacy reference '{needle}' in {path.relative_to(ROOT)}")


def check_ast_parse_references(failures: list[str]) -> None:
    for path in iter_text_files():
        if path.suffix not in {".py", ".cpp", ".h", ".md"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "ast.parse" not in text:
            continue
        if path not in ALLOWED_AST_PARSE_REFERENCES:
            failures.append(f"unexpected ast.parse reference in {path.relative_to(ROOT)}")


def check_python_c_api_references(failures: list[str]) -> None:
    for path in iter_text_files():
        if path.suffix not in {".c", ".cpp", ".h", ".hpp", ".txt"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "Python.h" not in text and "pybind11" not in text:
            continue
        if path in CPYTHON_SPEC_TEXT_FILES:
            continue
        failures.append(f"unexpected Python C API reference in {path.relative_to(ROOT)}")


def main() -> int:
    failures: list[str] = []
    check_removed_paths(failures)
    check_python_sources(failures)
    check_legacy_references(failures)
    check_ast_parse_references(failures)
    check_python_c_api_references(failures)
    if failures:
        print("frontend migration audit failed")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("frontend migration audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
