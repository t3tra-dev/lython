import ast

from ..mlir import ir
from ._base import BaseVisitor


class TypeIgnoreVisitor(BaseVisitor):
    """
    ```asdl
    type_ignore = TypeIgnore(int lineno, string tag)
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_type_ignore(self, node: ast.type_ignore) -> None:
        """
        ```asdl
        type_ignore = TypeIgnore(int lineno, string tag)
        ```
        """
        raise NotImplementedError("type_ignore not implemented")
