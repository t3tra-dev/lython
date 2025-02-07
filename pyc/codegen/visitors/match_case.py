import ast

from pyc.codegen.ir.builder import IRBuilder
from .base import BaseVisitor


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
