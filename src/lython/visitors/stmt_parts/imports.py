# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
from __future__ import annotations

import ast

from .._base import PRIMITIVE_BASE_TYPES


class StmtImportMixin:
    """Statement lowering for import-related AST nodes."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "asyncio":
                self._static_modules[alias.asname or alias.name] = "asyncio"
                continue
            raise NotImplementedError(f"Import '{alias.name}' not implemented")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module

        if module == "asyncio":
            supported = {"run", "create_task", "gather", "sleep"}
            for alias in node.names:
                name = alias.name
                if name not in supported:
                    raise NotImplementedError(f"Unknown asyncio import: {name}")
                self._static_module_symbols[alias.asname or name] = ("asyncio", name)
            return

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
