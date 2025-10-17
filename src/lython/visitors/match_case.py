import ast

from ..mlir import ir
from ._base import BaseVisitor


class MatchCaseVisitor(BaseVisitor):
    """
    ```asdl
    match_case = (pattern pattern, expr? guard, stmt* body)
    ```
    """

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_match_case(self, node: ast.match_case) -> None:
        """
        ```asdl
        match_case = (pattern pattern, expr? guard, stmt* body)
        ```
        """
        raise NotImplementedError("match_case not implemented")
