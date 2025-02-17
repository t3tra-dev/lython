import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class PatternVisitor(BaseVisitor):
    """
    ```asdl
    pattern = MatchValue(expr value)
            | MatchSingleton(constant value)
            | MatchSequence(pattern* patterns)
            | MatchMapping(expr* keys, pattern* patterns, identifier? rest)
            | MatchClass(expr cls, pattern* patterns, identifier* kwd_attrs, pattern* kwd_patterns)

            | MatchStar(identifier? name)
            -- The optional "rest" MatchMapping parameter handles capturing extra mapping keys

            | MatchAs(pattern? pattern, identifier? name)
            | MatchOr(pattern* patterns)

             attributes (int lineno, int col_offset, int end_lineno, int end_col_offset)
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_MatchValue(self, node: ast.MatchValue) -> None:
        """
        ```asdl
        MatchValue(expr value)
        ```
        """
        raise NotImplementedError("MatchValue not implemented")

    def visit_MatchSingleton(self, node: ast.MatchSingleton) -> None:
        """
        ```asdl
        MatchSingleton(constant value)
        ```
        """
        raise NotImplementedError("MatchSingleton not implemented")

    def visit_MatchSequence(self, node: ast.MatchSequence) -> None:
        """
        ```asdl
        MatchSequence(pattern* patterns)
        ```
        """
        raise NotImplementedError("MatchSequence not implemented")

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        """
        ```asdl
        MatchMapping(expr* keys, pattern* patterns, identifier? rest)
        ```
        """
        raise NotImplementedError("MatchMapping not implemented")

    def visit_MatchClass(self, node: ast.MatchClass) -> None:
        """
        ```asdl
        MatchClass(expr cls, pattern* patterns, identifier* kwd_attrs, pattern* kwd_patterns)
        ```
        """
        raise NotImplementedError("MatchClass not implemented")

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        """
        ```asdl
        MatchStar(identifier? name)
        ```
        """
        raise NotImplementedError("MatchStar not implemented")

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        """
        ```asdl
        MatchAs(pattern? pattern, identifier? name)
        ```
        """
        raise NotImplementedError("MatchAs not implemented")

    def visit_MatchOr(self, node: ast.MatchOr) -> None:
        """
        ```asdl
        MatchOr(pattern* patterns)
        ```
        """
        raise NotImplementedError("MatchOr not implemented")
