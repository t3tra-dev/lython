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

    ```asdl
    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit(self, node: ast.Compare, expr_visitor: ExprVisitor) -> TypedValue:
        """
        比較演算を処理する
        """
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single compare op is supported")

        left_typed = expr_visitor.visit(node.left)
        right_typed = expr_visitor.visit(node.comparators[0])

        if isinstance(node.ops[0], ast.Eq):
            return self.visit_Eq(left_typed, right_typed)
        elif isinstance(node.ops[0], ast.NotEq):
            return self.visit_NotEq(left_typed, right_typed)
        elif isinstance(node.ops[0], ast.Lt):
            return self.visit_Lt(left_typed, right_typed)
        elif isinstance(node.ops[0], ast.LtE):
            return self.visit_LtE(left_typed, right_typed)
        elif isinstance(node.ops[0], ast.Gt):
            return self.visit_Gt(left_typed, right_typed)
        elif isinstance(node.ops[0], ast.GtE):
            return self.visit_GtE(left_typed, right_typed)
        else:
            raise NotImplementedError(f"Unsupported compare op: {type(node.ops[0]).__name__}")

    def visit_Eq(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        Eq
        ```
        """
        tmp_bool = self.builder.get_temp_name()
        ret_val = self.builder.get_temp_name()
        self.builder.emit(f"  {tmp_bool} = icmp eq i32 {left.llvm_value}, {right.llvm_value}")
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")
        return TypedValue(ret_val, "i32")

    def visit_NotEq(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        NotEq
        ```
        """
        tmp_bool = self.builder.get_temp_name()
        ret_val = self.builder.get_temp_name()
        self.builder.emit(f"  {tmp_bool} = icmp ne i32 {left.llvm_value}, {right.llvm_value}")
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")
        return TypedValue(ret_val, "i32")

    def visit_Lt(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        Lt
        ```
        """
        tmp_bool = self.builder.get_temp_name()
        ret_val = self.builder.get_temp_name()
        self.builder.emit(f"  {tmp_bool} = icmp slt i32 {left.llvm_value}, {right.llvm_value}")
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")
        return TypedValue(ret_val, "i32")

    def visit_LtE(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        LtE
        ```
        """
        tmp_bool = self.builder.get_temp_name()
        ret_val = self.builder.get_temp_name()
        self.builder.emit(f"  {tmp_bool} = icmp sle i32 {left.llvm_value}, {right.llvm_value}")
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")
        return TypedValue(ret_val, "i32")

    def visit_Gt(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        Gt
        ```
        """
        tmp_bool = self.builder.get_temp_name()
        ret_val = self.builder.get_temp_name()
        self.builder.emit(f"  {tmp_bool} = icmp sgt i32 {left.llvm_value}, {right.llvm_value}")
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")
        return TypedValue(ret_val, "i32")

    def visit_GtE(self, left: TypedValue, right: TypedValue) -> TypedValue:
        """
        ```asdl
        GtE
        ```
        """
        tmp_bool = self.builder.get_temp_name()
        ret_val = self.builder.get_temp_name()
        self.builder.emit(f"  {tmp_bool} = icmp sge i32 {left.llvm_value}, {right.llvm_value}")
        self.builder.emit(f"  {ret_val} = zext i1 {tmp_bool} to i32")
        return TypedValue(ret_val, "i32")

    def visit_Is(self, node: ast.Is) -> str:
        """
        ```asdl
        Is
        ```
        """
        raise NotImplementedError("'is' operator not supported")

    def visit_IsNot(self, node: ast.IsNot) -> str:
        """
        ```asdl
        IsNot
        ```
        """
        raise NotImplementedError("'is not' operator not supported")

    def visit_In(self, node: ast.In) -> str:
        """
        ```asdl
        In
        ```
        """
        raise NotImplementedError("'in' operator not supported")

    def visit_NotIn(self, node: ast.NotIn) -> str:
        """
        ```asdl
        NotIn
        ```
        """
        raise NotImplementedError("'not in' operator not supported")

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unknown cmpop: {type(node).__name__}")
