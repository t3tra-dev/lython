import ast

from ..mlir import ir
from ._base import BaseVisitor


class TypeIgnoreVisitor(BaseVisitor):
    """
    ```asdl
    type_ignore = TypeIgnore(int lineno, string tag)
    ```
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def visit_TypeIgnore(self, node: ast.TypeIgnore) -> None:
        """
        ```asdl
        type_ignore = TypeIgnore(int lineno, string tag)
        ```
        """
        raise NotImplementedError("TypeIgnore not implemented")
