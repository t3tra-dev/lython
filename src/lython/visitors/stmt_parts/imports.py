# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
from __future__ import annotations

import ast

from .._base import PRIMITIVE_BASE_TYPES


class StmtImportMixin:
    """Statement lowering for import-related AST nodes."""

    def visit_Import(self, node: ast.Import) -> None:
        raise NotImplementedError("Import statement not implemented")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module

        if module == "lyrt":
            for alias in node.names:
                name = alias.name
                if name in ("native", "to_prim", "from_prim", "alloc", "dealloc"):
                    self._lyrt_builtins.add(name)
                else:
                    raise NotImplementedError(f"Unknown lyrt import: {name}")
            return

        if module == "lyrt.prim":
            valid_types = PRIMITIVE_BASE_TYPES | {"Vector", "Matrix", "Tensor"}
            for alias in node.names:
                name = alias.name
                if name in valid_types:
                    local_name = alias.asname or name
                    self._prim_types[local_name] = name
                else:
                    raise NotImplementedError(f"Unknown lyrt.prim type: {name}")
            return

        raise NotImplementedError(f"Import from '{module}' not implemented")
