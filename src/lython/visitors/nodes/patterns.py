from __future__ import annotations

import ast

from .._base import BaseVisitor

__all__ = ["PatternVisitor"]


class PatternVisitor(BaseVisitor):
    """
    ```asdl
    pattern = MatchValue(expr value)
            | MatchSingleton(constant value)
            | MatchSequence(pattern* patterns)
            | MatchMapping(expr* keys, pattern* patterns, identifier? rest)
            | MatchClass(expr cls, pattern* patterns, identifier* kwd_attrs,
                         pattern* kwd_patterns)
            | MatchStar(identifier? name)
            | MatchAs(pattern? pattern, identifier? name)
            | MatchOr(pattern* patterns)
    ```
    """

    def visit_MatchValue(self, node: ast.MatchValue) -> None:
        raise NotImplementedError("MatchValue not implemented")

    def visit_MatchSingleton(self, node: ast.MatchSingleton) -> None:
        raise NotImplementedError("MatchSingleton not implemented")

    def visit_MatchSequence(self, node: ast.MatchSequence) -> None:
        raise NotImplementedError("MatchSequence not implemented")

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        raise NotImplementedError("MatchMapping not implemented")

    def visit_MatchClass(self, node: ast.MatchClass) -> None:
        raise NotImplementedError("MatchClass not implemented")

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        raise NotImplementedError("MatchStar not implemented")

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        raise NotImplementedError("MatchAs not implemented")

    def visit_MatchOr(self, node: ast.MatchOr) -> None:
        raise NotImplementedError("MatchOr not implemented")
