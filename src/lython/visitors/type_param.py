import ast

from ..mlir import ir
from ._base import BaseVisitor


class TypeParamVisitor(BaseVisitor):
    """
    ```asdl
    type_param = TypeVar(identifier name, expr? bound, expr? default_value)
    ```
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def visit_type_param(self, node: ast.type_param) -> None:
        """
        ```asdl
        type_param = TypeVar(identifier name, expr? bound, expr? default_value)
        ```
        """
        raise NotImplementedError("type_param not implemented")
