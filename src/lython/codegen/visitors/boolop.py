import ast

from .base import BaseVisitor
from .expr import ExprVisitor
from ...codegen.ir.builder import IRBuilder


class BoolOpVisitor(BaseVisitor):
    """
    ```asdl
    boolop = And | Or
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit(self, node: ast.BoolOp, expr_visitor: ExprVisitor) -> str:
        """
        Determine the type of boolean operation and process it.
        """
        if len(node.values) != 2:
            raise NotImplementedError("Multi-value BoolOp not fully supported")

        left_val = expr_visitor.visit(node.values[0])
        right_val = expr_visitor.visit(node.values[1])

        if isinstance(node.op, ast.And):
            return self.visit_And(left_val, right_val)
        elif isinstance(node.op, ast.Or):
            return self.visit_Or(left_val, right_val)
        else:
            raise NotImplementedError(f"Unsupported BoolOp: {type(node.op).__name__}")

    def visit_And(self, left_val: str, right_val: str) -> str:
        """
        Process the 'And' boolean operation.

        ```asdl
        And
        ```
        """
        temp = self.builder.get_temp_name()
        self.builder.emit(f"  {temp} = and i32 {left_val}, {right_val}")
        return temp

    def visit_Or(self, left_val: str, right_val: str) -> str:
        """
        Process the 'Or' boolean operation.

        ```asdl
        Or
        ```
        """
        temp = self.builder.get_temp_name()
        self.builder.emit(f"  {temp} = or i32 {left_val}, {right_val}")
        return temp
