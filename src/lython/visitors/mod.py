from __future__ import annotations

import ast
from typing import Any

from ..mlir import ir
from ._base import BaseVisitor

__all__ = ["ModVisitor"]


class ModVisitor(BaseVisitor):
    """
    モジュール(ソースファイル全体)を訪問するクラス
    Python の ast.Module に対応

    ```asdl
    mod = Module(stmt* body, type_ignore* type_ignores)
        | Interactive(stmt* body)
        | Expression(expr body)
        | FunctionType(expr* argtypes, expr returns)
    ```
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def visit_Module(self, node: ast.Module) -> None:
        """
        ```asdl
        Module(stmt* body, type_ignore* type_ignores)
        ```
        """
        with ir.Location.unknown(self.ctx):
            module = ir.Module.create()
        self._set_module(module)

        for stmt in node.body:
            self.visit(stmt)
        return None

    def visit_Interactive(self, node: ast.Interactive) -> Any:
        """
        ```asdl
        Interactive(stmt* body)
        ```
        """
        raise NotImplementedError("Interactive mode not supported (static)")

    def visit_Expression(self, node: ast.Expression) -> Any:
        """
        ```asdl
        Expression(expr body)
        ```
        """
        raise NotImplementedError("Expression mode not supported")

    def visit_FunctionType(self, node: ast.FunctionType) -> None:
        """
        ```asdl
        FunctionType(expr* argtypes, expr returns)
        ```
        """
        raise NotImplementedError("Function type not supported in this static compiler")
