import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class MatchCaseVisitor(BaseVisitor):
    """
    ```asdl
    match_case = (pattern pattern, expr? guard, stmt* body)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_match_case(self, node: ast.match_case) -> None:
        """
        ```asdl
        match_case = (pattern pattern, expr? guard, stmt* body)
        ```
        """
        raise NotImplementedError("match_case not implemented")
