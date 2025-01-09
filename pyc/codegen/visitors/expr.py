from __future__ import annotations

import ast
from typing import Any, Dict, Callable

from ..ir import IRBuilder
from .base import BaseVisitor

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.counter = 0
        self.operations: Dict[type, Callable[[str, str], str]] = {
            ast.Add: lambda l, r: f"add i32 {l}, {r}",  # noqa
            ast.Sub: lambda l, r: f"sub i32 {l}, {r}",  # noqa
            ast.Mult: lambda l, r: f"mul i32 {l}, {r}",  # noqa
            ast.Div: lambda l, r: f"sdiv i32 {l}, {r}",  # noqa
        }

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        raise NotImplementedError("Boolean operation not supported")

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
                    result = self.visit_BinOp(node.args[0])
                    string_val = "%d\n"
                    str_ptr = self.builder.add_global_string(string_val)
                    self.builder.emit(
                        f"  %puts = call i32 @puts(i8* getelementptr inbounds ([{len(string_val) + 1} x i8], [{len(string_val) + 1} x i8]* {str_ptr}, i64 0, i64 0), i32 {result})"
                    )
            else:
                raise NotImplementedError("Only single argument printing is supported")
        else:
            raise NotImplementedError(f"Function {node.func.id} not supported")  # type: ignore

    def _get_operand(self, node: ast.AST) -> str:
        """オペランドの取得"""
        if isinstance(node, ast.Constant):
            return str(node.value)
        return self.visit(node)

    # 他のexpr関連のvisitメソッドも同様に実装
