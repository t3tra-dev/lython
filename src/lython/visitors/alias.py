import ast

from ..mlir import ir
from ._base import BaseVisitor


class AliasVisitor(BaseVisitor):
    """
    ```asdl
    -- import name with optional 'as' alias.
    alias = (identifier name, identifier? asname)
             attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_alias(self, node: ast.alias) -> None:
        """
        ```asdl
        -- import name with optional 'as' alias.
        alias = (
            identifier name,
            identifier? asname
        )
        ```
        """
        raise NotImplementedError("alias not implemeted")
