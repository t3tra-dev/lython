import ast

from ..mlir import ir
from ._base import BaseVisitor


class UnaryOpVisitor(BaseVisitor):
    """
    ```asdl
    unaryop = Invert | Not | UAdd | USub
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_Invert(self, node: ast.Invert) -> None:
        """
        ```asdl
        Invert
        ```
        """
        raise NotImplementedError("Invert unaryop not implemented")

    def visit_Not(self, node: ast.Not) -> None:
        """
        ```asdl
        Not
        ```
        """
        raise NotImplementedError("Not unaryop not implemented")

    def visit_UAdd(self, node: ast.UAdd) -> None:
        """
        ```asdl
        UAdd
        ```
        """
        raise NotImplementedError("UAdd unaryop not implemted")

    def visit_USub(self, node: ast.USub) -> None:
        """
        ```asdl
        USub
        ```
        """
        raise NotImplementedError("USub unaryop not implemted")
