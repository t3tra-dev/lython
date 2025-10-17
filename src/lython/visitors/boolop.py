import ast

from ..mlir import ir
from ._base import BaseVisitor

__all__ = ["BoolOpVisitor"]


class BoolOpVisitor(BaseVisitor):
    """
    論理演算子 (boolop) ノードを処理し、
    MLIR生成を行うクラス。

    ```asdl
    boolop = And | Or
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_BoolOp(self, node: ast.BoolOp) -> ir.Value:
        """
        論理演算子を処理する。
        短絡評価を適切に実装する。
        """
        raise NotImplementedError(f"Unsupported boolop: {type(node.op).__name__}")

    def visit_And(self, node: ast.And) -> ir.Value:
        """
        AND演算子 (論理積) を処理する。
        短絡評価: 左辺が偽なら右辺は評価しない。

        ```asdl
        And
        ```
        """
        raise NotImplementedError("And not implemented")

    def visit_Or(self, node: ast.Or) -> ir.Value:
        """
        OR演算子 (論理和) を処理する。
        短絡評価: 左辺が真なら右辺は評価しない。

        ```asdl
        Or
        ```
        """
        raise NotImplementedError("Or not implemented")
