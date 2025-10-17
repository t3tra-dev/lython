from __future__ import annotations

import ast

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

    def __init__(self, ctx: ir.Context) -> None:
        super().__init__(ctx)

    def visit_Eq(self, node: ast.Eq) -> None:
        """
        ```asdl
        Eq
        ```
        """
        raise NotImplementedError("'==' operator not supported")

    def visit_NotEq(self, node: ast.NotEq) -> None:
        """
        ```asdl
        NotEq
        ```
        """
        raise NotImplementedError("'!=' operator not supported")

    def visit_Lt(self, node: ast.Lt) -> None:
        """
        ```asdl
        Lt
        ```
        """
        raise NotImplementedError("'<' operator not supported")

    def visit_LtE(self, node: ast.LtE) -> None:
        """
        ```asdl
        LtE
        ```
        """
        raise NotImplementedError("'<=' operator not supported")

    def visit_Gt(self, node: ast.Gt) -> None:
        """
        ```asdl
        Gt
        ```
        """
        raise NotImplementedError("'>' operator not supported")

    def visit_GtE(self, node: ast.GtE) -> None:
        """
        ```asdl
        GtE
        ```
        """
        raise NotImplementedError("'>=' operator not supported")

    def visit_Is(self, node: ast.Is) -> None:
        """
        ```asdl
        Is
        ```
        """
        raise NotImplementedError("'is' operator not supported")

    def visit_IsNot(self, node: ast.IsNot) -> None:
        """
        ```asdl
        IsNot
        ```
        """
        raise NotImplementedError("'is not' operator not supported")

    def visit_In(self, node: ast.In) -> None:
        """
        ```asdl
        In
        ```
        """
        raise NotImplementedError("'in' operator not supported")

    def visit_NotIn(self, node: ast.NotIn) -> None:
        """
        ```asdl
        NotIn
        ```
        """
        raise NotImplementedError("'not in' operator not supported")
