from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor

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

    def generate_op(self, op_node: ast.AST, left_val: str, right_val: str) -> str:
        """
          - op_node: ast.Add / ast.Sub / ...
          - left_val, right_val: 左右オペランドのLLVM値 (str)
        ここで "add i32" 等のIRを生成し、結果変数を返す。
        """
        op_name = self.visit(op_node)  # 例: "Add", "Sub", ...
        result_name = self.builder.get_temp_name()

        if op_name == "Add":
            self.builder.emit(f"  {result_name} = add i32 {left_val}, {right_val}")
        elif op_name == "Sub":
            self.builder.emit(f"  {result_name} = sub i32 {left_val}, {right_val}")
        elif op_name == "Mult":
            self.builder.emit(f"  {result_name} = mul i32 {left_val}, {right_val}")
        elif op_name == "Div":
            self.builder.emit(f"  {result_name} = sdiv i32 {left_val}, {right_val}")
        elif op_name == "Mod":
            self.builder.emit(f"  {result_name} = srem i32 {left_val}, {right_val}")
        else:
            # TODO: 他演算子 (Pow, LShift, etc.) の処理を追加
            raise NotImplementedError(f"Operator '{op_name}' not supported yet")

        return result_name

    def visit_Add(self, node: ast.Add) -> str:
        return "Add"

    def visit_Sub(self, node: ast.Sub) -> str:
        return "Sub"

    def visit_Mult(self, node: ast.Mult) -> str:
        return "Mult"

    def visit_Div(self, node: ast.Div) -> str:
        return "Div"

    def visit_Mod(self, node: ast.Mod) -> str:
        return "Mod"

    def visit_BitOr(self, node: ast.BitOr) -> str:
        return "BitOr"

    def visit_BitAnd(self, node: ast.BitAnd) -> str:
        return "BitAnd"

    def visit_BitXor(self, node: ast.BitXor) -> str:
        return "BitXor"

    def visit_FloorDiv(self, node: ast.FloorDiv) -> str:
        return "FloorDiv"

    def visit_LShift(self, node: ast.LShift) -> str:
        return "LShift"

    def visit_RShift(self, node: ast.RShift) -> str:
        return "RShift"

    def visit_Pow(self, node: ast.Pow) -> str:
        return "Pow"

    def visit_MatMult(self, node: ast.MatMult) -> str:
        return "MatMult"

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unknown operator: {type(node).__name__}")
