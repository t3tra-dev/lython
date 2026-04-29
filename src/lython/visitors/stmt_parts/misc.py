# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast
from typing import Any


class StmtMiscMixin:
    """Statement lowering for simple or currently unsupported AST nodes."""

    def visit_Delete(self, node: ast.Delete) -> None:
        raise NotImplementedError("Delete statement not supported")

    def visit_Global(self, node: ast.Global) -> None:
        raise NotImplementedError("Global statement not implemented")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise NotImplementedError("Nonlocal statement not implemented")

    def visit_Expr(self, node: ast.Expr) -> Any:
        expr_visitor = self.subvisitors.get("Expr")
        if expr_visitor is None:
            raise NotImplementedError("Expression visitor not available")
        expr_visitor.current_block = self.current_block
        expr_visitor.visit(node.value)
        self.current_block = expr_visitor.current_block
        return None

    def visit_Pass(self, node: ast.Pass) -> None:
        raise NotImplementedError("Pass statement not implemented")

    def visit_Break(self, node: ast.Break) -> None:
        raise NotImplementedError("Break statement not implemented")

    def visit_Continue(self, node: ast.Continue) -> None:
        raise NotImplementedError("Continue statement not implemented")
