from __future__ import annotations

import ast
from typing import Callable, Dict

from ..ir.builder import IRBuilder
from .base import BaseVisitor

__all__ = ["ExprVisitor", "BinOpVisitor"]


class ExprVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.binop_visitor = BinOpVisitor(builder)

    def visit_Expr(self, node: ast.Expr) -> None:
        """式文の処理"""
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """関数呼び出しの処理"""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            if len(node.args) == 1:
                if isinstance(node.args[0], ast.Constant):
                    string_val = str(node.args[0].value)
                    str_ptr = self.builder.add_global_string(string_val)
                    self.builder.emit(
                        f"  %puts = call i32 @puts(i8* getelementptr inbounds ([{len(string_val) + 1} x i8], [{len(string_val) + 1} x i8]* {str_ptr}, i64 0, i64 0))"
                    )
                elif isinstance(node.args[0], ast.BinOp):
                    result = self.binop_visitor.visit(node.args[0])
                    string_val = "%d\n"
                    str_ptr = self.builder.add_global_string(string_val)
                    self.builder.emit(
                        f"  %puts = call i32 @puts(i8* getelementptr inbounds ([{len(string_val) + 1} x i8], [{len(string_val) + 1} x i8]* {str_ptr}, i64 0, i64 0), i32 {result})"
                    )
            else:
                raise NotImplementedError("Only single argument printing is supported")
        else:
            raise NotImplementedError(f"Function {node.func.id} not implemented")

    def visit_BinOp(self, node: ast.BinOp) -> str:
        return self.binop_visitor.visit(node)


class BinOpVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.counter = 0
        self.operations: Dict[type, Callable[[str, str], str]] = {
            ast.Add: lambda l, r: f"add i32 {l}, {r}",  # noqa
            ast.Sub: lambda l, r: f"sub i32 {l}, {r}",  # noqa
            ast.Mult: lambda l, r: f"mul i32 {l}, {r}",  # noqa
            ast.Div: lambda l, r: f"sdiv i32 {l}, {r}",  # noqa
        }

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """二項演算の処理"""
        # 左辺の処理
        if isinstance(node.left, ast.Constant):
            left = str(node.left.value)
        else:
            left = self.visit(node.left)

        # 右辺の処理
        if isinstance(node.right, ast.Constant):
            right = str(node.right.value)
        else:
            right = self.visit(node.right)

        # 演算の種類に応じた命令を生成
        operation = self.operations[type(node.op)]
        result = f"%{self.counter}"
        self.builder.emit(f"  {result} = {operation(left, right)}")
        self.counter += 1
        return result
