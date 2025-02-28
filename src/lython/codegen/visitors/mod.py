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

    ```asdl
    mod = Module(stmt* body, type_ignore* type_ignores)
        | Interactive(stmt* body)
        | Expression(expr body)
        | FunctionType(expr* argtypes, expr returns)
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_Module(self, node: ast.Module) -> None:
        """
        ファイル冒頭で、関数定義を先に処理
        main関数を生成し関数定義以外のトップレベル文を順次呼び出す

        ```asdl
        Module(stmt* body, type_ignore* type_ignores)
        ```
        """
        stmt_visitor: StmtVisitor = self.get_subvisitor("stmt")

        # 関数定義を先に処理
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                stmt_visitor.visit(stmt)

        # main関数を定義
        self.builder.emit("\ndefine i32 @main(i32 %argc, i8** %argv) {")
        self.builder.emit("entry:")

        # Boehm GCとオブジェクトシステムの初期化
        self.builder.emit("  call void @GC_init()")
        self.builder.emit("  call void @PyObject_InitSystem()")

        # 関数定義以外のステートメントを処理
        for stmt in node.body:
            if not isinstance(stmt, ast.FunctionDef):
                stmt_visitor.visit(stmt)

        # main関数終了
        self.builder.emit("  ret i32 0")
        self.builder.emit("}")

    def visit_Interactive(self, node: ast.Interactive) -> Any:
        """
        ```asdl
        Interactive(stmt* body)
        ```
        """
        raise NotImplementedError("Interactive mode not supported (static)")

    def visit_Expression(self, node: ast.Expression) -> Any:
        """
        ```asdl
        Expression(expr body)
        ```
        """
        raise NotImplementedError("Expression mode not supported")

    def visit_FunctionType(self, node: ast.FunctionType) -> None:
        """
        ```asdl
        FunctionType(expr* argtypes, expr returns)
        ```
        """
        raise NotImplementedError("Function type not supported in this static compiler")

    def generic_visit(self, node: ast.AST) -> None:
        """
        ModVisitor内で未対応ノードがあれば BaseVisitor にフォールバックさせる
        """
        return super().generic_visit(node)
