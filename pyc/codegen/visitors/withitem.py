import ast

from pyc.codegen.ir.builder import IRBuilder

from .base import BaseVisitor


class WithitemVisitor(BaseVisitor):
    """
    ```asdl
    withitem = (expr context_expr, expr? optional_vars)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_withitem(self, node: ast.withitem) -> None:
        """
        ```asdl
        withitem = (expr context_expr, expr? optional_vars)
        ```
        """
        raise NotImplementedError("withitem not implemented")
