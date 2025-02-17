import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class ExprContextVisitor(BaseVisitor):
    """
    ```asdl
    expr_context = Load | Store | Del
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

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
