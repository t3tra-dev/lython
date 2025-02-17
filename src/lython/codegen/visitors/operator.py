from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor, TypedValue

__all__ = ["OperatorVisitor"]


class OperatorVisitor(BaseVisitor):
    """
    演算子 (operator) ノードを処理し、
    LLVM IR生成を行うクラス。

    ```asdl
    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow
               | LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def generate_op(self, op_node: ast.AST, left: TypedValue, right: TypedValue) -> TypedValue:
        """
          - op_node: ast.Add / ast.Sub / ...
          - left_val, right_val: 左右オペランドのLLVM値 (str)
        ここで "add i32" 等のIRを生成し、結果変数を返す。
        """
        op_name = self.visit(op_node)  # "Add" / "Sub" / etc.
        result_name = self.builder.get_temp_name()

        # ここでは left/right ともに i32 前提
        assert left.type_ == "i32" and right.type_ == "i32", \
            f"Operator {op_name} expects i32, got {left.type_} and {right.type_}"

        lv = left.llvm_value
        rv = right.llvm_value

        if op_name == "Add":
            self.builder.emit(f"  {result_name} = add i32 {lv}, {rv}")
        elif op_name == "Sub":
            self.builder.emit(f"  {result_name} = sub i32 {lv}, {rv}")
        elif op_name == "Mult":
            self.builder.emit(f"  {result_name} = mul i32 {lv}, {rv}")
        elif op_name == "Div":
            self.builder.emit(f"  {result_name} = sdiv i32 {lv}, {rv}")
        elif op_name == "Mod":
            self.builder.emit(f"  {result_name} = srem i32 {lv}, {rv}")
        else:
            # TODO: 他演算子 (Pow, LShift, etc.) の処理を追加
            raise NotImplementedError(f"Operator '{op_name}' not supported yet")

        return TypedValue(result_name, "i32")

    def visit_Add(self, node: ast.Add) -> str:
        """
        ```asdl
        Add
        ```
        """
        return "Add"

    def visit_Sub(self, node: ast.Sub) -> str:
        """
        ```asdl
        Sub
        ```
        """
        return "Sub"

    def visit_Mult(self, node: ast.Mult) -> str:
        """
        ```asdl
        Mult
        ```
        """
        return "Mult"

    def visit_MatMult(self, node: ast.MatMult) -> str:
        """
        ```asdl
        MatMult
        ```
        """
        return "MatMult"

    def visit_Div(self, node: ast.Div) -> str:
        """
        ```asdl
        Div
        ```
        """
        return "Div"

    def visit_Mod(self, node: ast.Mod) -> str:
        """
        ```asdl
        Mod
        ```
        """
        return "Mod"

    def visit_Pow(self, node: ast.Pow) -> str:
        """
        ```asdl
        Pow
        ```
        """
        return "Pow"

    def visit_LShift(self, node: ast.LShift) -> str:
        """
        ```asdl
        LShift
        ```
        """
        return "LShift"

    def visit_RShift(self, node: ast.RShift) -> str:
        """
        ```asdl
        RShift
        ```
        """
        return "RShift"

    def visit_BitOr(self, node: ast.BitOr) -> str:
        """
        ```asdl
        BitOr
        ```
        """
        return "BitOr"

    def visit_BitXor(self, node: ast.BitXor) -> str:
        """
        ```asdl
        BitXor
        ```
        """
        return "BitXor"

    def visit_BitAnd(self, node: ast.BitAnd) -> str:
        """
        ```asdl
        BitAnd
        ```
        """
        return "BitAnd"

    def visit_FloorDiv(self, node: ast.FloorDiv) -> str:
        """
        ```asdl
        FloorDiv
        ```
        """
        return "FloorDiv"

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unknown operator: {type(node).__name__}")
