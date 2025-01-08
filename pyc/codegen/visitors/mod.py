import ast

from ..ir import IRBuilder
from .base import BaseVisitor
from .expr import ExprVisitor


class ModuleVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.expr_visitor = ExprVisitor(builder)

    def visit_Module(self, node: ast.Module) -> None:
        """モジュールの処理"""
        # 外部関数の宣言
        self.builder.emit("declare i32 @puts(i8* nocapture readonly) local_unnamed_addr")

        # main関数の開始
        self.builder.emit("\ndefine i32 @main(i32 %argc, i8** %argv) {")
        self.builder.emit("entry:")

        # 本体の処理
        for stmt in node.body:
            if isinstance(stmt, ast.Expr):
                self.expr_visitor.visit(stmt)
            else:
                self.visit(stmt)

        # 関数の終了
        self.builder.emit("  ret i32 0")
        self.builder.emit("}")
