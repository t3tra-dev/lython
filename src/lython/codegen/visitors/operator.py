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
        # 両方ともプリミティブ型の場合は直接計算
        if (
            not left.is_object
            and not right.is_object  # noqa
            and left.type_ == right.type_ # noqa
            and left.type_ in ("i32", "i1")  # noqa
        ):
            return self._generate_primitive_op(op_node, left, right)

        # オブジェクト演算の場合はメソッド呼び出し
        return self._generate_object_op(op_node, left, right)

    def _generate_primitive_op(self, op_node: ast.AST, left: TypedValue, right: TypedValue) -> TypedValue:
        """プリミティブ型同士の演算"""
        op_name = self.visit(op_node)  # "Add" / "Sub" / etc.
        result_name = self.builder.get_temp_name()

        # ここでは left/right ともに i32 前提
        assert left.type_ == "i32" and right.type_ == "i32", \
            f"Operator {op_name} expects i32, got {left.type_} and {right.type_}"

        # ここでは left/right ともに i32 前提
        assert left.type_ == "i32" and right.type_ == "i32", \
            f"Operator {op_name} expects i32, got {left.type_} and {right.type_}"

        lv = left.llvm_value
        rv = right.llvm_value

        if op_name == "Add":
            self.builder.emit(f"  {result_name} = add {left.type_} {lv}, {rv}")
        elif op_name == "Sub":
            self.builder.emit(f"  {result_name} = sub {left.type_} {lv}, {rv}")
        elif op_name == "Mult":
            self.builder.emit(f"  {result_name} = mul {left.type_} {lv}, {rv}")
        elif op_name == "Div":
            self.builder.emit(f"  {result_name} = sdiv {left.type_} {lv}, {rv}")
        elif op_name == "Mod":
            self.builder.emit(f"  {result_name} = srem {left.type_} {lv}, {rv}")
        elif op_name == "BitOr":
            self.builder.emit(f"  {result_name} = or {left.type_} {lv}, {rv}")
        elif op_name == "BitXor":
            self.builder.emit(f"  {result_name} = xor {left.type_} {lv}, {rv}")
        elif op_name == "BitAnd":
            self.builder.emit(f"  {result_name} = and {left.type_} {lv}, {rv}")
        elif op_name == "LShift":
            self.builder.emit(f"  {result_name} = shl {left.type_} {lv}, {rv}")
        elif op_name == "RShift":
            self.builder.emit(f"  {result_name} = ashr {left.type_} {lv}, {rv}")
        elif op_name == "FloorDiv":
            self.builder.emit(f"  {result_name} = sdiv {left.type_} {lv}, {rv}")
        else:
            raise NotImplementedError(f"Unknown cmpop '{op_name}'")

        return TypedValue.create_primitive(result_name, left.type_, left.python_type)

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
