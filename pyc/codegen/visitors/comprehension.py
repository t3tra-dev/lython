import ast

from pyc.codegen.ir.builder import IRBuilder
from .base import BaseVisitor


class ComprehensionVisitor(BaseVisitor):
    """
    ```asdl
    comprehension = (expr target, expr iter, expr* ifs, int is_async)
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
