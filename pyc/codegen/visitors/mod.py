from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor
from .stmt import StmtVisitor

__all__ = ["ModVisitor"]


class ModVisitor(BaseVisitor):
    """
    モジュール(ソースファイル全体)を訪問するクラス
    Python の ast.Module に対応
    """

    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.stmt_visitor = StmtVisitor(builder)

    def visit_Module(self, node: ast.Module) -> None:
        """
        ファイル冒頭で、関数定義を先に処理
        main関数を生成し関数定義以外のトップレベル文を順次呼び出す
        """
        # まず関数定義を走査
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.stmt_visitor.visit(stmt)

        # main関数を定義
        self.builder.emit("\ndefine i32 @main(i32 %argc, i8** %argv) {")
        self.builder.emit("entry:")

        # 関数定義以外のステートメントを処理
        for stmt in node.body:
            if not isinstance(stmt, ast.FunctionDef):
                self.stmt_visitor.visit(stmt)

        # main関数終了
        self.builder.emit("  ret i32 0")
        self.builder.emit("}")

    def visit_Interactive(self, node: ast.Interactive) -> Any:
        raise NotImplementedError("Interactive mode not supported (static)")

    def visit_Expression(self, node: ast.Expression) -> Any:
        raise NotImplementedError("Expression mode not supported")

    def visit_FunctionType(self, node: ast.FunctionType) -> None:
        raise NotImplementedError("Function type not supported in this static compiler")

    def generic_visit(self, node: ast.AST) -> None:
        super().generic_visit(node)
