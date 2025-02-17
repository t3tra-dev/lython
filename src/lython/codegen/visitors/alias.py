import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class AliasVisitor(BaseVisitor):
    """
    ```asdl
    -- import name with optional 'as' alias.
    alias = (identifier name, identifier? asname)
             attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

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
