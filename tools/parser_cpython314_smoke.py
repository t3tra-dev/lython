#!/usr/bin/env python3
"""Check CPython 3.14-only AST shapes produced by the C++ parser.

This is a verification harness, not a frontend dependency. It intentionally
does not call CPython's parser because the host Python in CI may be older than
the vendored CPython 3.14 grammar.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


class AstFactory(dict[str, object]):
    def __missing__(self, key: str) -> object:
        def construct(*args: object, **kwargs: object) -> dict[str, object]:
            return {"_": key, "__args__": list(args), **kwargs}

        self[key] = construct
        return construct


def parse_dump(text: str) -> object:
    env = AstFactory()
    env["Ellipsis"] = Ellipsis
    return eval(text, {"__builtins__": {}}, env)  # noqa: S307 - local AST dump


def lython_ast(lyc: Path, source: str) -> object:
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        Path(path).write_text(source)
        dump = subprocess.check_output(
            [str(lyc), "parse", "--attributes", path],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return parse_dump(dump.strip())
    finally:
        os.unlink(path)


def node_kind(value: object) -> str | None:
    if isinstance(value, dict):
        kind = value.get("_")
        if isinstance(kind, str):
            return kind
    return None


def nodes_by_kind(value: object, kind: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []

    def walk(item: object) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        if item.get("_") == kind:
            result.append(item)
        for child in item.values():
            walk(child)

    walk(value)
    return result


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def check_tstrings(root: object, failures: list[str]) -> None:
    template_strings = nodes_by_kind(root, "TemplateStr")
    interpolations = nodes_by_kind(root, "Interpolation")
    require(len(template_strings) >= 2, "expected at least two TemplateStr nodes", failures)
    require(len(interpolations) >= 3, "expected at least three Interpolation nodes", failures)

    numeric = next((node for node in interpolations if node.get("str") == "1"), None)
    require(numeric is not None, "missing t-string interpolation str='1'", failures)
    if numeric is not None:
        require(numeric.get("conversion") == ord("s"), "str='1' should use !s conversion", failures)
        require(
            node_kind(numeric.get("format_spec")) == "JoinedStr",
            "str='1' should carry JoinedStr format_spec",
            failures,
        )

    debug_name = next((node for node in interpolations if node.get("str") == "name"), None)
    require(debug_name is not None, "missing t-string debug interpolation str='name'", failures)
    if debug_name is not None:
        require(
            debug_name.get("conversion") == ord("r"),
            "debug t-string interpolation should default to !r",
            failures,
        )

    lambda_call = next(
        (node for node in interpolations if node.get("str") == "(lambda x: x)(1)"),
        None,
    )
    require(lambda_call is not None, "missing parenthesized lambda t-string interpolation", failures)
    if lambda_call is not None:
        require(
            node_kind(lambda_call.get("value")) == "Call",
            "parenthesized lambda interpolation should parse as Call",
            failures,
        )


def check_type_parameter_defaults(root: object, failures: list[str]) -> None:
    type_aliases = nodes_by_kind(root, "TypeAlias")
    type_vars = nodes_by_kind(root, "TypeVar")
    type_var_tuples = nodes_by_kind(root, "TypeVarTuple")
    param_specs = nodes_by_kind(root, "ParamSpec")

    require(type_aliases, "expected TypeAlias node", failures)
    require(type_vars, "expected TypeVar node", failures)
    require(type_var_tuples, "expected TypeVarTuple node", failures)
    require(param_specs, "expected ParamSpec node", failures)

    bounded_default = next((node for node in type_vars if node.get("name") == "T"), None)
    require(bounded_default is not None, "missing TypeVar T", failures)
    if bounded_default is not None:
        require(
            node_kind(bounded_default.get("bound")) == "Name",
            "TypeVar T should carry bound Name",
            failures,
        )
        require(
            node_kind(bounded_default.get("default_value")) == "Name",
            "TypeVar T should carry default_value Name",
            failures,
        )

    pack_default = next((node for node in type_var_tuples if node.get("name") == "Ts"), None)
    require(pack_default is not None, "missing TypeVarTuple Ts", failures)
    if pack_default is not None:
        require(
            node_kind(pack_default.get("default_value")) == "Starred",
            "TypeVarTuple Ts should carry Starred default_value",
            failures,
        )

    param_default = next((node for node in param_specs if node.get("name") == "P"), None)
    require(param_default is not None, "missing ParamSpec P", failures)
    if param_default is not None:
        require(
            node_kind(param_default.get("default_value")) == "List",
            "ParamSpec P should carry List default_value",
            failures,
        )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: parser_cpython314_smoke.py /path/to/lyc", file=sys.stderr)
        return 2
    lyc = Path(argv[1])
    source = "\n".join(
        [
            'text = t"v={1!s:>{width}} {name=}"',
            'value = t"{(lambda x: x)(1)}"',
            (
                "type Alias[T: int = str, *Ts = *tuple[int, ...], "
                "**P = [int, str]] = tuple[T, *Ts]"
            ),
            "",
        ]
    )
    root = lython_ast(lyc, source)
    failures: list[str] = []
    check_tstrings(root, failures)
    check_type_parameter_defaults(root, failures)
    if failures:
        print("CPython 3.14 parser smoke failed")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("CPython 3.14 parser smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
