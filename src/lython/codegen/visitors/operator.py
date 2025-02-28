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
    オブジェクトシステムと連携。

    ```asdl
    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow
               | LShift | RShift | BitOr | BitXor | BitAnd | FloorDiv
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

        # Pythonの算術演算子とそれに対応するメソッド名のマッピング
        self.binary_method_names = {
            ast.Add: "nb_add",
            ast.Sub: "nb_subtract",
            ast.Mult: "nb_multiply",
            ast.Div: "nb_true_divide",
            ast.Mod: "nb_remainder",
            ast.Pow: "nb_power",
            ast.LShift: "nb_lshift",
            ast.RShift: "nb_rshift",
            ast.BitOr: "nb_or",
            ast.BitXor: "nb_xor",
            ast.BitAnd: "nb_and",
            ast.FloorDiv: "nb_floor_divide",
        }

    def generate_op(self, op_node: ast.AST, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        二項演算子のIRコードを生成する
        - op_node: ast.Add / ast.Sub などの演算子ノード
        - left, right: 左右のオペランド
        """
        # 両方ともプリミティブ型の場合は直接計算（特にint型）
        if (
            not left.is_object and not right.is_object
            and left.python_type == "int" and right.python_type == "int"  # noqa
        ):
            return self._generate_direct_primitive_op(op_node, left, right)

        # 他のケースではオブジェクト演算
        return self._generate_object_op(op_node, left, right)

    def _generate_direct_primitive_op(self, op_node: ast.AST, left: TypedValue, right: TypedValue) -> TypedValue:
        """整数型の二項演算を直接LLVM命令で実装"""
        result = self.builder.get_temp_name()

        if isinstance(op_node, ast.Add):
            self.builder.emit(f"  {result} = add i32 {left.llvm_value}, {right.llvm_value}")
        elif isinstance(op_node, ast.Sub):
            self.builder.emit(f"  {result} = sub i32 {left.llvm_value}, {right.llvm_value}")
        elif isinstance(op_node, ast.Mult):
            self.builder.emit(f"  {result} = mul i32 {left.llvm_value}, {right.llvm_value}")
        elif isinstance(op_node, ast.Div) or isinstance(op_node, ast.FloorDiv):
            self.builder.emit(f"  {result} = sdiv i32 {left.llvm_value}, {right.llvm_value}")
        elif isinstance(op_node, ast.Mod):
            self.builder.emit(f"  {result} = srem i32 {left.llvm_value}, {right.llvm_value}")
        else:
            raise NotImplementedError(f"Unsupported primitive operation: {type(op_node).__name__}")

        return TypedValue.create_primitive(result, "i32", "int")

    def _generate_object_op(self, op_node: ast.AST, left: TypedValue, right: TypedValue) -> TypedValue:
        """オブジェクト同士の演算 (PyNumber_* メソッドを使用)"""
        # 両オペランドをオブジェクトに変換
        left_obj = self.ensure_object(left)
        right_obj = self.ensure_object(right)

        if not isinstance(op_node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv)):
            raise NotImplementedError(f"Unknown cmpop: {type(op_node).__name__}")

        method_name = self.binary_method_names.get(type(op_node))
        if method_name is None:
            raise NotImplementedError(f"Unknown cmpop: {type(op_node).__name__}")

        # PyNumber_* 関数を呼び出し
        result = self.builder.get_temp_name()

        # 演算子に応じた関数を選択
        if isinstance(op_node, ast.Add):
            self.builder.emit(f"  {result} = call ptr @PyNumber_Add(ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value})")
        elif isinstance(op_node, ast.Sub):
            self.builder.emit(f"  {result} = call ptr @PyNumber_Subtract(ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value})")
        elif isinstance(op_node, ast.Mult):
            self.builder.emit(f"  {result} = call ptr @PyNumber_Multiply(ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value})")
        elif isinstance(op_node, ast.Div):
            self.builder.emit(f"  {result} = call ptr @PyNumber_TrueDivide(ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value})")
        elif isinstance(op_node, ast.Mod):
            self.builder.emit(f"  {result} = call ptr @PyNumber_Remainder(ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value})")
        else:
            # 実際には個別の関数を呼び出すのが適切だが、
            # 現在は主要な演算子のみを実装
            self.builder.emit(f"  ; Unsupported cmpop: {type(op_node).__name__}")
            self.builder.emit(f"  {result} = call ptr @PyNumber_Add(ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value})")

        # 結果の型を決定（基本的には左オペランドの型を継承）
        return TypedValue.create_object(result, left.python_type)

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
