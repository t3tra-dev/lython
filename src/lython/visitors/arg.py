import ast

from ..mlir import ir
from ._base import BaseVisitor


class ArgVisitor(BaseVisitor):
    """
    ```asdl
    arg = (identifier arg, expr? annotation, string? type_comment)
           attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

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
