from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor
from .stmt import StmtVisitor

__all__ = ["ModVisitor"]


class ModVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.stmt_visitor = StmtVisitor(builder)

    def visit_Module(self, node: ast.Module) -> None:
        """モジュールの処理"""

        # main関数の開始
        self.builder.emit("\ndefine i32 @main(i32 %argc, i8** %argv) {")
        self.builder.emit("entry:")

        # 本体の処理
        for stmt in node.body:
            self.stmt_visitor.visit(stmt)

        # 関数の終了
        self.builder.emit("  ret i32 0")
        self.builder.emit("}")

    def visit_Interactive(self, node: ast.Interactive) -> None:
        """対話モードの処理"""
        raise NotImplementedError("Interactive mode not supported")

    def visit_Expression(self, node: ast.Expression) -> Any:
        """式の処理"""
        raise NotImplementedError("Expression mode not supported")

    def visit_FunctionType(self, node: ast.FunctionType) -> None:
        """関数型の処理"""
        raise NotImplementedError("Function type not supported")
