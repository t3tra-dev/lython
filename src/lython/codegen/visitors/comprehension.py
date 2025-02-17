import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class ComprehensionVisitor(BaseVisitor):
    """
    ```asdl
    comprehension = (expr target, expr iter, expr* ifs, int is_async)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """
        ```asdl
        comprehension = (expr target, expr iter, expr* ifs, int is_async)
        ```
        """
        raise NotImplementedError("Comprehension not implemented")
