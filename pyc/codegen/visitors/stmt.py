from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor
from .expr import ExprVisitor

__all__ = ["StmtVisitor"]


class StmtVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.expr_visitor = ExprVisitor(builder)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        raise NotImplementedError("Function definition not supported")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise NotImplementedError("Async function definition not supported")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise NotImplementedError("Class definition not supported")

    def visit_Return(self, node: ast.Return) -> None:
        raise NotImplementedError("Return statement not supported")

    def visit_Delete(self, node: ast.Delete) -> None:
        raise NotImplementedError("Delete statement not supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        raise NotImplementedError("Assignment not supported")

    def visit_Expr(self, node: ast.Expr) -> Any:
        """式文の処理"""
        return self.expr_visitor.visit(node.value)

    # 他のstmt関連のvisitメソッドも同様に実装
