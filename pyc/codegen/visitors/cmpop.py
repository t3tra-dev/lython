from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor

__all__ = ["CmpOpVisitor"]


class CmpOpVisitor(BaseVisitor):
    """
    比較演算子 (cmpop) ノードを処理し、
    LLVM IR生成を行うクラス。

    ```asdl
    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def generate_cmpop(self, op_node: ast.AST, left_val: str, right_val: str) -> str:
        """
          - op_node: ast.Eq / ast.Lt / ...
          - left_val, right_val: i32同士を想定
        ここで icmp eq/sl 等を生成し、i1 -> i32 に拡張して結果を返す。
        """
        op_name = self.visit(op_node)  # "Eq", "Lt", etc.
        tmp_bool = self.builder.get_temp_name()

        # i1 -> i32 zext分
        ret_val = self.builder.get_temp_name()

        if op_name == "Eq":
            self.builder.emit(f"  {tmp_bool} = icmp eq i32 {left_val}, {right_val}")
        elif op_name == "NotEq":
            self.builder.emit(f"  {tmp_bool} = icmp ne i32 {left_val}, {right_val}")
        elif op_name == "Lt":
            self.builder.emit(f"  {tmp_bool} = icmp slt i32 {left_val}, {right_val}")
        elif op_name == "LtE":
            self.builder.emit(f"  {tmp_bool} = icmp sle i32 {left_val}, {right_val}")
        elif op_name == "Gt":
            self.builder.emit(f"  {tmp_bool} = icmp sgt i32 {left_val}, {right_val}")
        elif op_name == "GtE":
            self.builder.emit(f"  {tmp_bool} = icmp sge i32 {left_val}, {right_val}")
        else:
            raise NotImplementedError(f"Comparison operator '{op_name}' not implemented")

        # i1 -> i32 に zext
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")
        return ret_val

    def visit_Eq(self, node: ast.Eq) -> str:
        """
        ```asdl
        Eq
        ```
        """
        return "Eq"

    def visit_NotEq(self, node: ast.NotEq) -> str:
        """
        ```asdl
        NotEq
        ```
        """
        return "NotEq"

    def visit_Lt(self, node: ast.Lt) -> str:
        """
        ```asdl
        Lt
        ```
        """
        return "Lt"

    def visit_LtE(self, node: ast.LtE) -> str:
        """
        ```asdl
        LtE
        ```
        """
        return "LtE"

    def visit_Gt(self, node: ast.Gt) -> str:
        """
        ```asdl
        Gt
        ```
        """
        return "Gt"

    def visit_GtE(self, node: ast.GtE) -> str:
        """
        ```asdl
        GtE
        ```
        """
        return "GtE"

    def visit_Is(self, node: ast.Is) -> str:
        """
        ```asdl
        Is
        ```
        """
        return "Is"

    def visit_IsNot(self, node: ast.IsNot) -> str:
        """
        ```asdl
        IsNot
        ```
        """
        return "IsNot"

    def visit_In(self, node: ast.In) -> str:
        """
        ```asdl
        In
        ```
        """
        return "In"

    def visit_NotIn(self, node: ast.NotIn) -> str:
        """
        ```asdl
        NotIn
        ```
        """
        return "NotIn"

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unknown cmpop: {type(node).__name__}")
