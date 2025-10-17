from __future__ import annotations

import ast

from ..mlir import ir
from ._base import BaseVisitor

__all__ = ["OperatorVisitor"]


class OperatorVisitor(BaseVisitor):
    """
    演算子 (operator) ノードを処理し、
    MLIR生成を行うクラス。

    ```asdl
    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow
               | LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_Add(self, node: ast.Add) -> None:
        """
        ```asdl
        Add
        ```
        """
        raise NotImplementedError("Add operator not implemented")

    def visit_Sub(self, node: ast.Sub) -> None:
        """
        ```asdl
        Sub
        ```
        """
        raise NotImplementedError("Sub operator not implemented")

    def visit_Mult(self, node: ast.Mult) -> None:
        """
        ```asdl
        Mult
        ```
        """
        raise NotImplementedError("Mult operator not implemented")

    def visit_MatMult(self, node: ast.MatMult) -> None:
        """
        ```asdl
        MatMult
        ```
        """
        raise NotImplementedError("MatMult operator not implemented")

    def visit_Div(self, node: ast.Div) -> None:
        """
        ```asdl
        Div
        ```
        """
        raise NotImplementedError("Div operator not implemented")

    def visit_Mod(self, node: ast.Mod) -> None:
        """
        ```asdl
        Mod
        ```
        """
        raise NotImplementedError("Mod operator not implemented")

    def visit_Pow(self, node: ast.Pow) -> None:
        """
        ```asdl
        Pow
        ```
        """
        raise NotImplementedError("Pow operator not implemented")

    def visit_LShift(self, node: ast.LShift) -> None:
        """
        ```asdl
        LShift
        ```
        """
        raise NotImplementedError("LShift operator not implemented")

    def visit_RShift(self, node: ast.RShift) -> None:
        """
        ```asdl
        RShift
        ```
        """
        raise NotImplementedError("RShift operator not implemented")

    def visit_BitOr(self, node: ast.BitOr) -> None:
        """
        ```asdl
        BitOr
        ```
        """
        raise NotImplementedError("BitOr operator not implemented")

    def visit_BitXor(self, node: ast.BitXor) -> None:
        """
        ```asdl
        BitXor
        ```
        """
        raise NotImplementedError("BitXor operator not implemented")

    def visit_BitAnd(self, node: ast.BitAnd) -> None:
        """
        ```asdl
        BitAnd
        ```
        """
        raise NotImplementedError("BitAnd operator not implemented")

    def visit_FloorDiv(self, node: ast.FloorDiv) -> None:
        """
        ```asdl
        FloorDiv
        ```
        """
        raise NotImplementedError("FloorDiv operator not implemented")
