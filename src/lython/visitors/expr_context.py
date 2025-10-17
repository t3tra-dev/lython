import ast

from ..mlir import ir
from ._base import BaseVisitor


class ExprContextVisitor(BaseVisitor):
    """
    ```asdl
    expr_context = Load | Store | Del
    ```
    """

    def __init__(self, ctx: ir.Context) -> None:
        super().__init__(ctx)

    def visit_Load(self, node: ast.Load) -> None:
        """
        ```asdl
        Load
        ```
        """
        raise NotImplementedError("Load expression context not implemented")

    def visit_Store(self, node: ast.Store) -> None:
        """
        ```asdl
        Store
        ```
        """
        raise NotImplementedError("Store expression context not implemented")

    def visit_Del(self, node: ast.Del) -> None:
        """
        ```asdl
        Del
        ```
        """
        raise NotImplementedError("Del expression context not implemented")
