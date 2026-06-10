#!/usr/bin/env python3
"""Compare the C++ parser dump against CPython ast for shared syntax.

This is a verification harness, not a frontend dependency.  It intentionally
uses only syntax accepted by the Python interpreter running CI so that the
vendored CPython 3.14 parser can be checked against CPython AST semantics where
the grammars overlap.
"""

from __future__ import annotations

import ast
import os
import pprint
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class AstFactory(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        def construct(*args: object, **kwargs: object) -> dict[str, object]:
            return {"_": key, "__args__": list(args), **kwargs}

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
    if kind == "arguments":
        result: dict[str, object] = {"_": "arguments"}
        for field in (
            "posonlyargs",
            "args",
            "vararg",
            "kwonlyargs",
            "kw_defaults",
            "kwarg",
            "defaults",
        ):
            normalized = normalize(value.get(field))
            if normalized is None or normalized == []:
                continue
            result[field] = normalized
        return result

    result = {"_": kind} if kind else {}
    for field, raw in value.items():
        if field in {"_", "__args__"}:
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


def cpython_ast(source: str, mode: str) -> object:
    tree = ast.parse(source, mode=mode, type_comments=True)
    return normalize(parse_dump(ast.dump(tree, include_attributes=True)))


def lython_ast(lyc: Path, source: str, mode: str) -> object:
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        Path(path).write_text(source)
        command = [
            str(lyc),
            "parse",
            "--type-comments",
            "--attributes",
        ]
        if mode == "eval":
            command.append("--expression")
        elif mode == "single":
            command.append("--interactive")
        elif mode == "func_type":
            command.append("--function-type")
        command.append(path)
        dump = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        )
        return normalize(parse_dump(dump.strip()))
    finally:
        os.unlink(path)


MODULE_SNIPPETS = [
    'x = "a" "b" f"{c}" "d"\n',
    'x = f"{c}" "a"\n',
    "x = 1; y = 2;\n",
    "if x: y = 1\nelse: y = 2\n",
    "x = f(\n    a,\n    b=1,\n)\n",
    "x = [\n    a,\n    b,\n]\n",
    "x = a + \\\n    b\n",
    "@dec(\n    arg\n)\ndef f():\n    pass\n",
    "x = a and b and c\ny = a or b or c\nz = (a or b) and (c or d)\n",
    "x = []  # type: list[int]\nfor item in xs:  # type: int\n    pass\n",
    "match x:\n    case C(a, b=2):\n        pass\n",
    "async def f(xs):\n    async for x in xs:\n        await g(x)\n",
    "def g(xs):\n    yield\n    yield from xs\n    x = (yield 1)\n",
    "try:\n    pass\nexcept (A, B) as e:\n    pass\n",
    "match x:\n    case {1: a, 2: b, **rest}:\n        pass\n",
    "match x:\n    case [a, b, *rest]:\n        pass\n",
    "x = a[1, 2:3, ..., ::-1]\n",
    "def f(a, /, b: int = 1, *args: str, c, d=2, **kw) -> None:\n    pass\n",
    "@dec\nclass C(Base, metaclass=M):\n    pass\n",
    "import a as b, c.d\nfrom ...pkg import x as y, z\n",
    "try:\n    a()\nexcept A:\n    b()\nelse:\n    c()\nfinally:\n    d()\n",
    "with (cm1 as a, cm2 as b):\n    pass\n",
    (
        "x = [a + b for a in xs for b in ys if a if b]\n"
        "y = {a: b for a, b in pairs}\n"
        "z = {a for a in xs}\n"
        "w = (a for a in xs)\n"
    ),
    "f = lambda x, /, y=1, *args, z, **kw: x + y\n",
    'x = {**a, "b": 1, **c}\ny = {*a, 1, *b}\n',
    "match x:\n    case mod.VALUE | -1 | 1 + 2j:\n        pass\n",
    "match x:\n    case C(attr=_ as y):\n        pass\n",
    'x = b"a" b"\\x62"\ny = r"\\n"\nz = "\\N{LATIN CAPITAL LETTER A}"\n',
    "x = (a := 1)\ny = [a := item for item in xs]\n",
    "del obj.attr, xs[0], name\n",
    'assert x, "message"\nraise RuntimeError("x") from None\n',
    'x: int\ny: int = 1\nobj.attr: str = "s"\nxs[0]: float = 1.0\n',
    (
        "x += 1\n"
        "y -= 2\n"
        "z *= 3\n"
        "a @= b\n"
        "c //= d\n"
        "e <<= f\n"
        "g >>= h\n"
        "i &= j\n"
        "k ^= l\n"
        "m |= n\n"
    ),
    "x = +a\ny = ~b\nz = not c\nw = a @ b / c % d // e << f >> g ** h\n",
    "x = a == b != c < d <= e > f >= g is h is not i in j not in k\n",
    (
        "while x:\n"
        "    if y:\n"
        "        continue\n"
        "    break\n"
        "else:\n"
        "    pass\n"
    ),
    (
        "x = None\n"
        "y = True\n"
        "z = False\n"
        "match x:\n"
        "    case None:\n"
        "        pass\n"
        "    case True:\n"
        "        pass\n"
        "    case False:\n"
        "        pass\n"
    ),
    "try:\n    a()\nexcept* A as e:\n    b(e)\n",
    "async def f():\n    async with cm as value:\n        return value\n",
    (
        "def outer():\n"
        "    global g\n"
        "    nonlocal_missing = 0\n"
        "    def inner():\n"
        "        nonlocal nonlocal_missing\n"
        "        return nonlocal_missing\n"
        "    return inner\n"
    ),
    "def f(x):\n    return x\n",
    "# type: ignore[attr-defined]\nx = 1  # type: ignore\n",
    "type Alias[T, *Ts, **P] = tuple[T, *Ts]\n",
]

MODE_SNIPPETS = [
    ("exec", source) for source in MODULE_SNIPPETS
] + [
    ("eval", "1\n"),
    ("eval", "1, 2\n"),
    ("eval", "a if b else c\n"),
    ("eval", "lambda x, /, y=1: x + y\n"),
    ("eval", "[x for x in xs if x]\n"),
    ("eval", "{x: y for x, y in pairs}\n"),
    ("eval", 'f"{x=!r:>{w}}"\n'),
    ("eval", "a[1, 2:3, ..., ::-1]\n"),
    ("single", "pass\n"),
    ("single", "x = 1\n"),
    ("single", "if x:\n    y = 1\n"),
    ("single", "def f():\n    pass\n"),
    ("single", "match x:\n    case 1:\n        pass\n"),
    ("func_type", "(int) -> str\n"),
    ("func_type", "(int, str) -> bool\n"),
    ("func_type", "(*tuple[int, ...], **dict[str, int]) -> tuple[int, ...]\n"),
    ("func_type", "() -> None\n"),
]

EXPECTED_SHARED_NODE_KINDS = {
    "Add",
    "And",
    "AnnAssign",
    "Assert",
    "Assign",
    "AsyncFor",
    "AsyncFunctionDef",
    "AsyncWith",
    "Attribute",
    "AugAssign",
    "Await",
    "BinOp",
    "BitAnd",
    "BitOr",
    "BitXor",
    "BoolOp",
    "Break",
    "Call",
    "ClassDef",
    "Compare",
    "Constant",
    "Continue",
    "Del",
    "Delete",
    "Dict",
    "DictComp",
    "Div",
    "Eq",
    "ExceptHandler",
    "Expr",
    "Expression",
    "FloorDiv",
    "For",
    "FormattedValue",
    "FunctionDef",
    "FunctionType",
    "GeneratorExp",
    "Global",
    "Gt",
    "GtE",
    "If",
    "IfExp",
    "Import",
    "ImportFrom",
    "In",
    "Interactive",
    "Invert",
    "Is",
    "IsNot",
    "JoinedStr",
    "LShift",
    "Lambda",
    "List",
    "ListComp",
    "Load",
    "Lt",
    "LtE",
    "MatMult",
    "Match",
    "MatchAs",
    "MatchClass",
    "MatchMapping",
    "MatchOr",
    "MatchSequence",
    "MatchSingleton",
    "MatchStar",
    "MatchValue",
    "Mod",
    "Module",
    "Mult",
    "Name",
    "NamedExpr",
    "Nonlocal",
    "Not",
    "NotEq",
    "NotIn",
    "Or",
    "ParamSpec",
    "Pass",
    "Pow",
    "RShift",
    "Raise",
    "Return",
    "Set",
    "SetComp",
    "Slice",
    "Starred",
    "Store",
    "Sub",
    "Subscript",
    "Try",
    "TryStar",
    "Tuple",
    "TypeAlias",
    "TypeIgnore",
    "TypeVar",
    "TypeVarTuple",
    "UAdd",
    "USub",
    "UnaryOp",
    "While",
    "With",
    "Yield",
    "YieldFrom",
    "alias",
    "arg",
    "arguments",
    "comprehension",
    "keyword",
    "match_case",
    "withitem",
}

CPYTHON_314_ONLY_NODE_KINDS = {
    "Interpolation",
    "TemplateStr",
}


def cpython_asdl_node_kinds() -> set[str]:
    asdl_path = Path(__file__).resolve().parents[1] / "src/lython/parser/Python.asdl"
    sum_types = {
        "mod",
        "stmt",
        "expr",
        "expr_context",
        "boolop",
        "operator",
        "unaryop",
        "cmpop",
        "pattern",
        "type_param",
    }
    product_types = {
        "comprehension",
        "arguments",
        "arg",
        "keyword",
        "alias",
        "withitem",
        "match_case",
    }

    result: set[str] = set()
    current: str | None = None
    for line in asdl_path.read_text().splitlines():
        stripped = line.split("--")[0].strip()
        if not stripped:
            continue
        definition = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", stripped)
        if definition:
            current = definition.group(1)
            rhs = definition.group(2)
            if current in product_types and rhs.startswith("("):
                result.add(current)
                continue
            if current in sum_types or current in {"excepthandler", "type_ignore"}:
                for part in rhs.split("|"):
                    name = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", part.strip())
                    if name:
                        result.add(name.group(1))
                continue
        elif stripped.startswith("|") and current in sum_types:
            for part in stripped[1:].split("|"):
                name = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", part.strip())
                if name:
                    result.add(name.group(1))
    return result


def collect_node_kinds(value: object, result: set[str]) -> None:
    if isinstance(value, list):
        for item in value:
            collect_node_kinds(item, result)
        return
    if not isinstance(value, dict):
        return
    kind = value.get("_")
    if isinstance(kind, str):
        result.add(kind)
    for item in value.values():
        collect_node_kinds(item, result)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: parser_ast_parity_smoke.py /path/to/lyc", file=sys.stderr)
        return 2
    lyc = Path(argv[1])
    failures: list[tuple[str, str, object, object]] = []
    covered_node_kinds: set[str] = set()
    for mode, source in MODE_SNIPPETS:
        expected = cpython_ast(source, mode)
        actual = lython_ast(lyc, source, mode)
        collect_node_kinds(actual, covered_node_kinds)
        if expected != actual:
            failures.append((mode, source, expected, actual))

    if failures:
        print(f"CPython AST parity smoke failed ({len(failures)} snippets)")
        for mode, source, expected, actual in failures:
            print(f"\n--- source ({mode}) ---")
            print(source)
            print("--- CPython ---")
            pprint.pp(expected, width=180)
            print("--- Lython ---")
            pprint.pp(actual, width=180)
        return 1

    missing_node_kinds = EXPECTED_SHARED_NODE_KINDS - covered_node_kinds
    if missing_node_kinds:
        print("CPython AST parity smoke coverage failed")
        print("missing node kinds:", ", ".join(sorted(missing_node_kinds)))
        return 1

    declared_kinds = EXPECTED_SHARED_NODE_KINDS | CPYTHON_314_ONLY_NODE_KINDS
    asdl_kinds = cpython_asdl_node_kinds()
    missing_declared = asdl_kinds - declared_kinds
    extra_declared = declared_kinds - asdl_kinds
    if missing_declared or extra_declared:
        print("CPython AST parity smoke ASDL coverage declaration failed")
        if missing_declared:
            print("missing declared kinds:", ", ".join(sorted(missing_declared)))
        if extra_declared:
            print("extra declared kinds:", ", ".join(sorted(extra_declared)))
        return 1

    print(f"CPython AST parity smoke passed ({len(MODE_SNIPPETS)} snippets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
