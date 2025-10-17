import ast

from ..mlir import ir
from ._base import BaseVisitor


class ComprehensionVisitor(BaseVisitor):
    """
    ```asdl
    comprehension = (expr target, expr iter, expr* ifs, int is_async)
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """
        ```asdl
        comprehension = (expr target, expr iter, expr* ifs, int is_async)
        ```
        """
        raise NotImplementedError("Comprehension not implemented")
