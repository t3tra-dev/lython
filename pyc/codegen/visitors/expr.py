import ast

from ..ir.builder import IRBuilder
from .base import BaseVisitor

__all__ = ["ExpressionVisitor"]


class ExpressionVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder

    def visit_Expr(self, node: ast.Expr) -> None:
        """式文の処理"""
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """関数呼び出しの処理"""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                string_val = node.args[0].value
                str_ptr = self.builder.add_global_string(string_val)
                self.builder.emit(
                    f"  %puts = call i32 @puts(i8* getelementptr inbounds ([{len(string_val) + 1} x i8], [{len(string_val) + 1} x i8]* {str_ptr}, i64 0, i64 0))"
                )
            else:
                raise NotImplementedError("Only simple string printing is supported")
        else:
            raise NotImplementedError(f"Function {node.func.id} not implemented")
