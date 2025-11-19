import ast

from ..mlir import ir
from ._base import BaseVisitor


class WithitemVisitor(BaseVisitor):
    """
    ```asdl
    withitem = (expr context_expr, expr? optional_vars)
    ```
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def visit_withitem(self, node: ast.withitem) -> None:
        """
        ```asdl
        withitem = (expr context_expr, expr? optional_vars)
        ```
        """
        raise NotImplementedError("withitem not implemented")
