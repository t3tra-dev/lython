from __future__ import annotations

import ast
from typing import Any

from ..mlir import ir
from ._base import BaseVisitor

__all__ = ["CmpOpVisitor"]


class CmpOpVisitor(BaseVisitor):
    """
    比較演算子 (cmpop) ノードを処理し、
    MLIR生成を行うクラス。

    ```asdl
    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_Compare(self, node: ast.Compare) -> ir.Value:
        raise NotImplementedError("Compare node handling not implemented")

    def visit_Eq(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """
        ```asdl
        Eq
        ```
        """
        raise NotImplementedError("'==' operator not supported")

    def visit_NotEq(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """
        ```asdl
        NotEq
        ```
        """
        raise NotImplementedError("'!=' operator not supported")

    def visit_Lt(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """
        ```asdl
        Lt
        ```
        """
        raise NotImplementedError("'<' operator not supported")

    def visit_LtE(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """
        ```asdl
        LtE
        ```
        """
        raise NotImplementedError("'<=' operator not supported")

    def visit_Gt(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """
        ```asdl
        Gt
        ```
        """
        raise NotImplementedError("'>' operator not supported")

    def visit_GtE(self, left: ir.Value, right: ir.Value) -> ir.Value:
        """
        ```asdl
        GtE
        ```
        """
        raise NotImplementedError("'>=' operator not supported")

    def visit_Is(self, node: ast.Is) -> ir.Value:
        """
        ```asdl
        Is
        ```
        """
        raise NotImplementedError("'is' operator not supported")

    def visit_IsNot(self, node: ast.IsNot) -> ir.Value:
        """
        ```asdl
        IsNot
        ```
        """
        raise NotImplementedError("'is not' operator not supported")

    def visit_In(self, node: ast.In) -> ir.Value:
        """
        ```asdl
        In
        ```
        """
        raise NotImplementedError("'in' operator not supported")

    def visit_NotIn(self, node: ast.NotIn) -> ir.Value:
        """
        ```asdl
        NotIn
        ```
        """
        raise NotImplementedError("'not in' operator not supported")

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unknown cmpop: {type(node).__name__}")
