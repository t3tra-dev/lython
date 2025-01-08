import ast
from typing import Any

__all__ = ["BaseVisitor"]


class BaseVisitor:
    def visit(self, node: ast.AST) -> Any:
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> None:
        raise NotImplementedError(f"Node type {type(node).__name__} not implemented")
