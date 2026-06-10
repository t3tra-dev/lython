#!/usr/bin/env python3
"""Check CPython-style source decoding for the self-hosted parser.

The parser itself must not depend on CPython. This smoke uses the host
interpreter only as an oracle for source bytes that overlap with the supported
encoding subset.
"""

from __future__ import annotations

import ast
import os
import pprint
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class AstFactory(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        def construct(*args: object, **kwargs: object) -> dict[str, object]:
            return {"_": key, "args": list(args), **kwargs}

        self[key] = construct
        return construct


def parse_dump(text: str) -> object:
    env = AstFactory()
    env["Ellipsis"] = Ellipsis
    return eval(text, {"__builtins__": {}}, env)  # noqa: S307 - local AST dump


def normalize(value: object) -> object:
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if not isinstance(value, dict):
        return value

    kind = value.get("_")
    result = {"_": kind} if kind else {}
    for field, raw in value.items():
        if field in {"_", "args"}:
            continue
        normalized = normalize(raw)
        if normalized is None or normalized == []:
            continue
        if field == "kind" and normalized is None:
            continue
        if field == "type_comment" and normalized is None:
            continue
        result[field] = normalized
    return result


def cpython_ast(source: bytes) -> object:
    tree = compile(source, "<source-encoding-smoke>", "exec", ast.PyCF_ONLY_AST)
    return normalize(parse_dump(ast.dump(tree, include_attributes=True)))


def lython_ast(lyc: Path, source: bytes) -> object:
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        Path(path).write_bytes(source)
        dump = subprocess.check_output(
            [str(lyc), "parse", "--attributes", path],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return normalize(parse_dump(dump.strip()))
    finally:
        os.unlink(path)


def lython_rejects(lyc: Path, source: bytes) -> bool:
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        Path(path).write_bytes(source)
        completed = subprocess.run(
            [str(lyc), "parse", path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.returncode != 0
    finally:
        os.unlink(path)


VALID_CASES = [
    b'x = "\xe2\x9c\x93"\n',
    b'\xef\xbb\xbfx = "bom"\n',
    b"# coding: ascii\nx = 'ascii'\n",
    b"# coding: latin-1\nx = '\xe9'\n",
    b"#!/usr/bin/env lython\n# coding=iso-8859-1\nx = '\xe9'\n",
]

INVALID_CASES = [
    b"# coding: ascii\nx = '\xc3\xa9'\n",
    b"\xef\xbb\xbf# coding: latin-1\nx = 'bad bom'\n",
]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: parser_source_encoding_smoke.py /path/to/lyc", file=sys.stderr)
        return 2

    lyc = Path(argv[1])
    failures: list[tuple[str, bytes, object, object]] = []
    for source in VALID_CASES:
        expected = cpython_ast(source)
        actual = lython_ast(lyc, source)
        if expected != actual:
            failures.append(("accepted", source, expected, actual))

    for source in INVALID_CASES:
        try:
            compile(source, "<source-encoding-smoke>", "exec", ast.PyCF_ONLY_AST)
            expected_reject = False
        except SyntaxError:
            expected_reject = True
        actual_reject = lython_rejects(lyc, source)
        if expected_reject != actual_reject:
            failures.append(("rejected", source, expected_reject, actual_reject))

    if failures:
        print(f"CPython source encoding smoke failed ({len(failures)} cases)")
        for mode, source, expected, actual in failures:
            print(f"\n--- source ({mode}) ---")
            print(source)
            print("--- CPython ---")
            pprint.pp(expected, width=180)
            print("--- Lython ---")
            pprint.pp(actual, width=180)
        return 1

    print(
        "CPython source encoding smoke passed "
        f"({len(VALID_CASES)} accepted, {len(INVALID_CASES)} rejected)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
