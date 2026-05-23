from __future__ import annotations

import ast


class _UnsupportedFinallyControlDetector(ast.NodeVisitor):
    def __init__(
        self, *, allow_raise: bool = False, allow_return: bool = False
    ) -> None:
        self.allow_raise = allow_raise
        self.allow_return = allow_return
        self.unsupported: str | None = None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        if self.allow_return:
            return
        self.unsupported = "return"

    def visit_Raise(self, node: ast.Raise) -> None:
        if self.allow_raise:
            return
        self.unsupported = "raise"

    def visit_Break(self, node: ast.Break) -> None:
        self.unsupported = "break"

    def visit_Continue(self, node: ast.Continue) -> None:
        self.unsupported = "continue"


def find_unsupported_finally_control(
    nodes: list[ast.stmt], *, allow_raise: bool = False, allow_return: bool = False
) -> str | None:
    detector = _UnsupportedFinallyControlDetector(
        allow_raise=allow_raise, allow_return=allow_return
    )
    for node in nodes:
        detector.visit(node)
        if detector.unsupported is not None:
            return detector.unsupported
    return None


class _FinallyReturnDetector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.found = True


def contains_return(nodes: list[ast.stmt]) -> bool:
    detector = _FinallyReturnDetector()
    for node in nodes:
        detector.visit(node)
        if detector.found:
            return True
    return False


def stmt_always_returns(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Return, ast.Raise)):
        return True
    if isinstance(node, ast.If):
        return always_returns(node.body) and always_returns(node.orelse)
    if isinstance(node, ast.Try):
        return always_returns(node.finalbody) or (
            always_returns(node.body)
            and bool(node.handlers)
            and all(always_returns(handler.body) for handler in node.handlers)
            and (not node.orelse or always_returns(node.orelse))
        )
    return False


def always_returns(nodes: list[ast.stmt]) -> bool:
    return bool(nodes) and stmt_always_returns(nodes[-1])


__all__ = [
    "always_returns",
    "contains_return",
    "find_unsupported_finally_control",
    "stmt_always_returns",
]
