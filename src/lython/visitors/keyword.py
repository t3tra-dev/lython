import ast

from ..mlir import ir
from ._base import BaseVisitor


class KeywordVisitor(BaseVisitor):
    """
    ```asdl
    -- keyword arguments supplied to call (NULL identifier for **kwargs)
    keyword = (identifier? arg, expr value)
               attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """

    def __init__(self, ctx: ir.Context) -> None:
        super().__init__(ctx)

    def visit_keyword(self, node: ast.keyword) -> None:
        """
        ```asdl
        keyword = (
            identifier? arg,
            expr value
        )
        ```
        """
        raise NotImplementedError("keyword not implemented")
