import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class ArgVisitor(BaseVisitor):
    """
    ```asdl
    arg = (identifier arg, expr? annotation, string? type_comment)
           attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_arg(self, node: ast.arg) -> None:
        """
        ```asdl
        arg = (
            identifier arg,
            expr? annotation,
            string? type_comment
        )
        ```
        """
        raise NotImplementedError("arg not implemented")
