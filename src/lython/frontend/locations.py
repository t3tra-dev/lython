from __future__ import annotations

import ast


def source_position(node: ast.AST) -> tuple[int, int] | None:
    """Return 1-based line and 0-based column when an AST node carries a source span."""
    try:
        line = object.__getattribute__(node, "lineno")
        col = object.__getattribute__(node, "col_offset")
    except AttributeError:
        return None
    if not isinstance(line, int) or not isinstance(col, int):
        return None
    return line, col
