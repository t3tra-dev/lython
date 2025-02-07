import ast

from pyc.codegen.ir.builder import IRBuilder
from .base import BaseVisitor


class BoolOpVisitor(BaseVisitor):
    """
    ```asdl
    boolop = And | Or
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_And(self, node: ast.And) -> None:
        """
        ```asdl
        And
        ```
        """
        raise NotImplementedError("And boolop not implemented")

    def visit_Or(self, node: ast.Or) -> None:
        """
        ```asdl
        Or
        ```
        """
        raise NotImplementedError("Or boolop not implemented")
