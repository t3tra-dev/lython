import ast

from pyc.codegen.ir.builder import IRBuilder
from .base import BaseVisitor


class ExceptHandlerVisitor(BaseVisitor):
    """
    ```asdl
    excepthandler = ExceptHandler(expr? type, identifier? name, stmt* body)
                    attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """
        ```asdl
        ExceptHandler(expr? type, identifier? name, stmt* body)
        ```
        """
        raise NotImplementedError("ExceptHandler not implemented")
