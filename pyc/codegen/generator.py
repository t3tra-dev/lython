import ast

from .ir.builder import IRBuilder
from .visitors import ModVisitor, ExprVisitor
from .visitors.base import BaseVisitor

__all__ = ["IRGenerator"]


class IRGenerator(BaseVisitor):
    def __init__(self):
        self.builder = IRBuilder()
        self.module_visitor = ModVisitor(self.builder)
        self.expr_visitor = ExprVisitor(self.builder)

    def visit_Module(self, node: ast.Module) -> None:
        self.module_visitor.visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.expr_visitor.visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.expr_visitor.visit_Call(node)

    def generate(self, node: ast.AST) -> str:
        self.visit(node)
        return "\n".join(list(self.builder.global_strings.values()) + self.builder.output)
