from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor, TypedValue
from .expr import ExprVisitor

__all__ = ["CmpOpVisitor"]


class CmpOpVisitor(BaseVisitor):
    """
    比較演算子 (cmpop) ノードを処理し、
    LLVM IR生成を行うクラス。
    オブジェクトシステムと連携して PyObject_RichCompare を使用。

    ```asdl
    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)
        # 比較演算子のPython定数へのマッピング
        self.op_to_const = {
            ast.Eq: 2,     # Py_EQ
            ast.NotEq: 3,  # Py_NE
            ast.Lt: 0,     # Py_LT
            ast.LtE: 1,    # Py_LE
            ast.Gt: 4,     # Py_GT
            ast.GtE: 5,    # Py_GE
        }

    def visit(self, node: ast.Compare, expr_visitor: ExprVisitor) -> TypedValue:
        """
        比較演算を処理する
        """
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single compare op is supported")

        # 左右のオペランドを評価
        left_typed = expr_visitor.visit(node.left)
        right_typed = expr_visitor.visit(node.comparators[0])
        op_type = type(node.ops[0])

        # 演算子に対応する定数を取得
        op_type = type(node.ops[0])
        if op_type not in self.op_to_const:
            raise NotImplementedError(f"Unknown cmpop: {op_type.__name__}")
        op_const = self.op_to_const[op_type]
        if op_const is None:
            raise NotImplementedError(f"Unknown cmpop: {op_type.__name__}")

        # プリミティブ型同士の比較かどうかを確認
        if (
            not left_typed.is_object
            and not right_typed.is_object  # noqa
            and left_typed.type_ == right_typed.type_  # noqa
            and left_typed.type_ in ("i32", "i1")  # noqa
        ):
            # プリミティブ型の直接比較
            return self._compare_primitives(left_typed, right_typed, op_type)
        else:
            # オブジェクト比較 (PyObject_RichCompare を使用)
            return self._compare_objects(left_typed, right_typed, op_const)

    def _compare_primitives(self, left: TypedValue, right: TypedValue, op_type) -> TypedValue:
        """プリミティブ型同士の比較"""
        tmp_bool = self.builder.get_temp_name()

        # 整数型の場合は直接比較で最適化
        if left.type_ == "i32" and right.type_ == "i32":
            if op_type == ast.Eq:
                self.builder.emit(f"  {tmp_bool} = icmp eq i32 {left.llvm_value}, {right.llvm_value}")
            elif op_type == ast.NotEq:
                self.builder.emit(f"  {tmp_bool} = icmp ne i32 {left.llvm_value}, {right.llvm_value}")
            elif op_type == ast.Lt:
                self.builder.emit(f"  {tmp_bool} = icmp slt i32 {left.llvm_value}, {right.llvm_value}")
            elif op_type == ast.LtE:
                self.builder.emit(f"  {tmp_bool} = icmp sle i32 {left.llvm_value}, {right.llvm_value}")
            elif op_type == ast.Gt:
                self.builder.emit(f"  {tmp_bool} = icmp sgt i32 {left.llvm_value}, {right.llvm_value}")
            elif op_type == ast.GtE:
                self.builder.emit(f"  {tmp_bool} = icmp sge i32 {left.llvm_value}, {right.llvm_value}")

            # i1を返す（直接使用可能なブール値）
            return TypedValue.create_primitive(tmp_bool, "i1", "bool")

        if op_type == ast.Eq:
            self.builder.emit(f"  {tmp_bool} = icmp eq {left.type_} {left.llvm_value}, {right.llvm_value}")
        elif op_type == ast.NotEq:
            self.builder.emit(f"  {tmp_bool} = icmp ne {left.type_} {left.llvm_value}, {right.llvm_value}")
        elif op_type == ast.Lt:
            self.builder.emit(f"  {tmp_bool} = icmp slt {left.type_} {left.llvm_value}, {right.llvm_value}")
        elif op_type == ast.LtE:
            self.builder.emit(f"  {tmp_bool} = icmp sle {left.type_} {left.llvm_value}, {right.llvm_value}")
        elif op_type == ast.Gt:
            self.builder.emit(f"  {tmp_bool} = icmp sgt {left.type_} {left.llvm_value}, {right.llvm_value}")
        elif op_type == ast.GtE:
            self.builder.emit(f"  {tmp_bool} = icmp sge {left.type_} {left.llvm_value}, {right.llvm_value}")

        # i1をi32に拡張
        ret_val = self.builder.get_temp_name()
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")

        return TypedValue.create_primitive(ret_val, "i32", "bool")

    def _compare_objects(self, left: TypedValue, right: TypedValue, op_const: int) -> TypedValue:
        """オブジェクト同士の比較 (PyObject_RichCompare を使用)"""
        # 両オペランドをオブジェクトに変換
        left_obj = self.ensure_object(left)
        right_obj = self.ensure_object(right)

        # PyObject_RichCompare を呼び出し
        result = self.builder.get_temp_name()
        self.builder.emit(f"  {result} = call ptr @PyObject_RichCompare(ptr {left_obj.llvm_value}, ptr {right_obj.llvm_value}, i32 {op_const})")

        return TypedValue.create_object(result, "bool")

    def visit_Eq(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        Eq
        ```
        """
        return self._compare_objects(left, right, 2)  # Py_EQ

    def visit_NotEq(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        NotEq
        ```
        """
        return self._compare_objects(left, right, 3)  # Py_NE

    def visit_Lt(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        Lt
        ```
        """
        return self._compare_objects(left, right, 0)  # Py_LT

    def visit_LtE(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        LtE
        ```
        """
        return self._compare_objects(left, right, 1)  # Py_LE

    def visit_Gt(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        Gt
        ```
        """
        return self._compare_objects(left, right, 4)  # Py_GT

    def visit_GtE(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        GtE
        ```
        """
        return self._compare_objects(left, right, 5)  # Py_GE

    def visit_Is(self, node: ast.Is) -> TypedValue:
        """
        ```asdl
        Is
        ```
        """
        raise NotImplementedError("'is' operator not supported")

    def visit_IsNot(self, node: ast.IsNot) -> TypedValue:
        """
        ```asdl
        IsNot
        ```
        """
        raise NotImplementedError("'is not' operator not supported")

    def visit_In(self, node: ast.In) -> TypedValue:
        """
        ```asdl
        In
        ```
        """
        raise NotImplementedError("'in' operator not supported")

    def visit_NotIn(self, node: ast.NotIn) -> TypedValue:
        """
        ```asdl
        NotIn
        ```
        """
        raise NotImplementedError("'not in' operator not supported")

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unknown cmpop: {type(node).__name__}")
